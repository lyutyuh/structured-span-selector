import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from bert_modelling import BertModel
from transformers import   BertTokenizer

import logging
import numpy as np
from collections import Iterable, defaultdict

from outside_mp import CFGMentionProposer
from greedy_mp import GreedyMentionProposer

from util import logsumexp, log1mexp, batch_select, bucket_distance


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger()


class CorefModel(torch.nn.Module):
    tz = BertTokenizer.from_pretrained("bert-base-cased")
    def __init__(self, config, device, num_genres=None):
        super().__init__()
        self.config = config
        self.device = device
        
        self.cls_id = self.tz.convert_tokens_to_ids("[CLS]")
        self.sep_id = self.tz.convert_tokens_to_ids("[SEP]")
        
        self.num_genres = num_genres if num_genres else len(config['genres'])
        self.max_seg_len = config['max_segment_len']
        self.max_span_width = config['max_span_width']

        # Model
        self.dropout = nn.Dropout(p=config['dropout_rate'])
        self.bert = BertModel.from_pretrained(config['bert_pretrained_name_or_path'])

        self.bert_emb_size = self.bert.config.hidden_size
        self.span_emb_size = self.bert_emb_size * 3
        if config['use_features']:
            self.span_emb_size += config['feature_emb_size']
        
        self.pair_emb_size = self.span_emb_size * 3
        if config['use_metadata']:
            self.pair_emb_size += 2 * config['feature_emb_size']
        if config['use_features']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_segment_distance']:
            self.pair_emb_size += config['feature_emb_size']
            
        if config['mention_proposer'].lower() == "outside":
            self.mention_proposer = CFGMentionProposer(max_span_width=self.max_span_width, neg_sample_rate=config['neg_sample_rate'])
        elif config['mention_proposer'].lower() == "greedy":
            self.mention_proposer = GreedyMentionProposer()

        self.emb_span_width = self.make_embedding(self.max_span_width) if config['use_features'] else None
        self.emb_span_width_prior = self.make_embedding(self.max_span_width) if config['use_width_prior'] else None
        self.emb_antecedent_distance_prior = self.make_embedding(10) if config['use_distance_prior'] else None
        self.emb_genre = self.make_embedding(self.num_genres)
        self.emb_same_speaker = self.make_embedding(2) if config['use_metadata'] else None
        self.emb_segment_distance = self.make_embedding(config['max_training_sentences']) if config['use_segment_distance'] else None
        self.emb_top_antecedent_distance = self.make_embedding(10)

        self.mention_token_attn = self.make_ffnn(self.bert_emb_size, 0, output_size=1) if config['model_heads'] else None
        if type(self.mention_proposer) == CFGMentionProposer:
            self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'], output_size=2)
        elif type(self.mention_proposer) == GreedyMentionProposer:
            self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'], output_size=output_size)
                
        self.span_width_score_ffnn = self.make_ffnn(config['feature_emb_size'], [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if config['use_width_prior'] else None
        self.coarse_bilinear = self.make_ffnn(self.span_emb_size, 0, output_size=self.span_emb_size)
        self.antecedent_distance_score_ffnn = self.make_ffnn(config['feature_emb_size'], 0, output_size=1) if config['use_distance_prior'] else None
        self.coref_score_ffnn = self.make_ffnn(self.pair_emb_size, [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if config['fine_grained'] else None

        self.all_words = 0
        self.all_pred_men = 0
        self.debug = False
        

    def make_embedding(self, dict_size, std=0.02):
        emb = nn.Embedding(dict_size, self.config['feature_emb_size'])
        init.normal_(emb.weight, std=std)
        return emb

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i-1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def get_params(self, named=False):
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith('bert') or name.startswith("mention_transformer"):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    def forward(self, *input):
        mention_doc = self.get_mention_doc(*input)
        return self.get_predictions_and_loss(mention_doc, *input)
    
    def get_flat_span_location_indices(self, spans, sentence_map):
        sentence_map = sentence_map.tolist()
        spans_list = spans.tolist()
        flat_span_location_indices = []
        
        prev_sent_id = sentence_map[0]
        sentence_lengths, cur_sent_len = [], 0
        for i in sentence_map:
            if prev_sent_id == i:
                cur_sent_len += 1
            else:
                sentence_lengths.append(cur_sent_len)
                cur_sent_len = 1
                prev_sent_id = i
        
        sentence_lengths.append(cur_sent_len)
        max_sentence_len = max(sentence_lengths)
        
        sentence_offsets = np.cumsum([0] + sentence_lengths)[:-1]
        for (start, end) in spans_list:
            sent_id = sentence_map[start] - sentence_map[0]
            offset = sentence_offsets[sent_id]
            
            flat_id = sent_id * (max_sentence_len**2) + (start-offset)*max_sentence_len + (end-offset)
            flat_span_location_indices.append(flat_id)
            
        sentence_lengths = spans.new_tensor(sentence_lengths)
        flat_span_location_indices = spans.new_tensor(flat_span_location_indices)
        return flat_span_location_indices, sentence_lengths
    
    def get_mention_doc(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                                 is_training, gold_starts=None, gold_ends=None, gold_mention_cluster_map=None, 
                                 coreferable_starts=None, coreferable_ends=None, 
                                 constituent_starts=None, constituent_ends=None, constituent_type=None):
        
        mention_doc = self.bert(input_ids, attention_mask=input_mask)  # [num seg, num max tokens, emb size]
        mention_doc = mention_doc["last_hidden_state"]
        input_mask = input_mask.bool()
        mention_doc = mention_doc[input_mask]
        return mention_doc
        
    
    def get_predictions_and_loss(
        self, mention_doc, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
        is_training, gold_starts=None, gold_ends=None, gold_mention_cluster_map=None, 
        coreferable_starts=None, coreferable_ends=None, 
        constituent_starts=None, constituent_ends=None, constituent_type=None
    ):
        """ Model and input are already on the device """
        device = self.device
        conf = self.config

        do_loss = False
        if gold_mention_cluster_map is not None:
            assert gold_starts is not None
            assert gold_ends is not None
            do_loss = True
            
        input_mask = input_mask.bool()
        speaker_ids = speaker_ids[input_mask]
        num_words = mention_doc.shape[0]
        
        self.all_words += num_words
        
        # Get candidate span
        sentence_indices = sentence_map  # [num tokens]
        candidate_starts = torch.unsqueeze(torch.arange(0, num_words, device=device), 1).repeat(1, self.max_span_width)
        candidate_ends = candidate_starts + torch.arange(0, self.max_span_width, device=device)
        candidate_start_sent_idx = sentence_indices[candidate_starts]
        candidate_end_sent_idx = sentence_indices[torch.min(candidate_ends, torch.tensor(num_words - 1, device=device))]
        candidate_mask = (candidate_ends < num_words) & (candidate_start_sent_idx == candidate_end_sent_idx)
        candidate_mask &= (input_ids[input_mask][candidate_starts] != self.cls_id)
        candidate_mask &= (input_ids[input_mask][torch.clamp(candidate_ends, max=num_words-1)] != self.sep_id)
        
        candidate_starts, candidate_ends = candidate_starts[candidate_mask], candidate_ends[candidate_mask]  # [num valid candidates]
        num_candidates = candidate_starts.shape[0]

        candidate_labels = None
        non_dummy_indicator = None
        # Get candidate labels
        if do_loss:
            same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(candidate_starts, 0))
            same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(candidate_ends, 0))
            same_span = (same_start & same_end).long()
            candidate_labels = torch.matmul(gold_mention_cluster_map.unsqueeze(0).type_as(mention_doc), same_span.type_as(mention_doc))
            candidate_labels = candidate_labels.long().squeeze()  # [num candidates]; non-gold span has label 0
            
            
        # Get span embedding
        span_start_emb, span_end_emb = mention_doc[candidate_starts], mention_doc[candidate_ends]
        # span_start_emb_1, span_end_emb_1 = mention_doc[candidate_starts], mention_doc[candidate_ends+1]
        # candidate_emb_list = [span_start_emb, span_end_emb]
        candidate_emb_list = [span_start_emb, span_end_emb]
        if conf['use_features']:
            candidate_width_idx = candidate_ends - candidate_starts
            candidate_width_emb = self.emb_span_width(candidate_width_idx)
            candidate_width_emb = self.dropout(candidate_width_emb)
            candidate_emb_list.append(candidate_width_emb)
        # Use attended head or avg token
        candidate_tokens = torch.unsqueeze(torch.arange(0, num_words, device=device), 0).repeat(num_candidates, 1)
        candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidate_starts, 1)) & (candidate_tokens <= torch.unsqueeze(candidate_ends, 1))
        if conf['model_heads']:
            token_attn = self.mention_token_attn(mention_doc).squeeze()
        else:
            token_attn = torch.ones(num_words, dtype=mention_doc.dtype, device=device)  # Use avg if no attention
        candidate_tokens_attn_raw = candidate_tokens_mask.log() + token_attn.unsqueeze(0)
        candidate_tokens_attn = F.softmax(candidate_tokens_attn_raw, dim=1)
        
        head_attn_emb = torch.matmul(candidate_tokens_attn, mention_doc)
        candidate_emb_list.append(head_attn_emb)
        candidate_span_emb = torch.cat(candidate_emb_list, dim=-1)  # [num candidates, new emb size]

        # Get span scores
        candidate_mention_scores_and_parsing = self.span_emb_score_ffnn(candidate_span_emb)
        
        if type(self.mention_proposer) == CFGMentionProposer:
            candidate_mention_scores, candidate_mention_parsing_scores = candidate_mention_scores_and_parsing.split(1, dim=-1)
            candidate_mention_scores = candidate_mention_scores.squeeze(1)
        elif type(self.mention_proposer) == GreedyMentionProposer:
            candidate_mention_scores = candidate_mention_scores_and_parsing.squeeze(-1)
            candidate_mention_parsing_scores = candidate_mention_scores
            
        if conf['use_width_prior']:
            width_score = self.span_width_score_ffnn(self.emb_span_width_prior.weight).squeeze(1)
            candidate_mention_scores = candidate_mention_scores + width_score[candidate_width_idx]
        
            
        spans = torch.stack([candidate_starts, candidate_ends], dim=-1)
        flat_span_location_indices, sentence_lengths = self.get_flat_span_location_indices(
            spans, sentence_map
        )
        num_top_spans = int(min(conf['max_num_extracted_spans'], conf['top_span_ratio'] * num_words))
        non_dummy_indicator = (candidate_labels > 0) if candidate_labels is not None else None
        
        if type(self.mention_proposer) == CFGMentionProposer:
            top_span_p_mention, selected_idx, top_spans, mp_loss, _ = self.mention_proposer(
                spans,
                candidate_mention_parsing_scores, 
                candidate_mask[candidate_mask],
                non_dummy_indicator if non_dummy_indicator is not None else None,
                sentence_lengths,
                num_top_spans,
                flat_span_location_indices,
            )
            top_span_log_p_mention = top_span_p_mention.log()
            top_span_log_p_mention = top_span_log_p_mention[selected_idx]
            
        elif type(self.mention_proposer) == GreedyMentionProposer:
            _, selected_idx, top_spans, mp_loss, _ = self.mention_proposer(
                spans, 
                candidate_mention_parsing_scores, 
                candidate_mask[candidate_mask],
                sentence_lengths,
                num_top_spans,
            )
        
        num_top_spans = selected_idx.size(0)

        top_span_starts, top_span_ends = candidate_starts[selected_idx], candidate_ends[selected_idx]
        top_span_emb = candidate_span_emb[selected_idx]
        top_span_cluster_ids = candidate_labels[selected_idx] if do_loss else None
        top_span_mention_scores = candidate_mention_scores[selected_idx]
        
        # Coarse pruning on each mention's antecedents
        max_top_antecedents = min(num_top_spans, conf['max_top_antecedents'])
        top_span_range = torch.arange(0, num_top_spans, device=device)
        antecedent_offsets = torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(top_span_range, 0)
        antecedent_mask = (antecedent_offsets >= 1)
        pairwise_mention_score_sum = torch.unsqueeze(top_span_mention_scores, 1) + torch.unsqueeze(top_span_mention_scores, 0)
        source_span_emb = self.dropout(self.coarse_bilinear(top_span_emb))
        target_span_emb = self.dropout(torch.transpose(top_span_emb, 0, 1))
        pairwise_coref_scores = torch.matmul(source_span_emb, target_span_emb)
        
        pairwise_fast_scores = pairwise_mention_score_sum + pairwise_coref_scores
        pairwise_fast_scores += antecedent_mask.type_as(mention_doc).log()
        if conf['use_distance_prior']:
            distance_score = torch.squeeze(self.antecedent_distance_score_ffnn(self.dropout(self.emb_antecedent_distance_prior.weight)), 1)
            bucketed_distance = bucket_distance(antecedent_offsets)
            antecedent_distance_score = distance_score[bucketed_distance]
            pairwise_fast_scores += antecedent_distance_score
        # Slow mention ranking
        if conf['fine_grained']:
            top_pairwise_fast_scores, top_antecedent_idx = torch.topk(pairwise_fast_scores, k=max_top_antecedents)
            top_antecedent_mask = batch_select(antecedent_mask, top_antecedent_idx, device)  # [num top spans, max top antecedents]
            top_antecedent_offsets = batch_select(antecedent_offsets, top_antecedent_idx, device)
            
            same_speaker_emb, genre_emb, seg_distance_emb, top_antecedent_distance_emb = None, None, None, None
            if conf['use_metadata']:
                top_span_speaker_ids = speaker_ids[top_span_starts]
                top_antecedent_speaker_id = top_span_speaker_ids[top_antecedent_idx]
                same_speaker = torch.unsqueeze(top_span_speaker_ids, 1) == top_antecedent_speaker_id
                same_speaker_emb = self.emb_same_speaker(same_speaker.long())
                genre_emb = self.emb_genre(genre)
                genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat(num_top_spans, max_top_antecedents, 1)
            if conf['use_segment_distance']:
                num_segs, seg_len = input_ids.shape[0], input_ids.shape[1]
                token_seg_ids = torch.arange(0, num_segs, device=device).unsqueeze(1).repeat(1, seg_len)
                token_seg_ids = token_seg_ids[input_mask]
                top_span_seg_ids = token_seg_ids[top_span_starts]
                top_antecedent_seg_ids = token_seg_ids[top_span_starts[top_antecedent_idx]]
                top_antecedent_seg_distance = torch.unsqueeze(top_span_seg_ids, 1) - top_antecedent_seg_ids
                top_antecedent_seg_distance = torch.clamp(top_antecedent_seg_distance, 0, self.config['max_training_sentences'] - 1)
                seg_distance_emb = self.emb_segment_distance(top_antecedent_seg_distance)
            if conf['use_features']:  # Antecedent distance
                top_antecedent_distance = bucket_distance(top_antecedent_offsets)
                top_antecedent_distance_emb = self.emb_top_antecedent_distance(top_antecedent_distance)

            top_antecedent_emb = top_span_emb[top_antecedent_idx]  # [num top spans, max top antecedents, emb size]
            feature_list = []
            if conf['use_metadata']:  # speaker, genre
                feature_list.append(same_speaker_emb)
                feature_list.append(genre_emb)
            if conf['use_segment_distance']:
                feature_list.append(seg_distance_emb)
            if conf['use_features']:  # Antecedent distance
                feature_list.append(top_antecedent_distance_emb)
            feature_emb = torch.cat(feature_list, dim=2)
            feature_emb = self.dropout(feature_emb)
            target_emb = torch.unsqueeze(top_span_emb, 1).repeat(1, max_top_antecedents, 1)
            # target_parent_emb = torch.unsqueeze(top_span_parent_emb, 1).repeat(1, max_top_antecedents, 1)
            similarity_emb = target_emb * top_antecedent_emb
            pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)
            top_pairwise_slow_scores = self.coref_score_ffnn(pair_emb).squeeze(2)
            # print(pair_emb.size(), mention_doc.size(),  pair_emb.size(0) / mention_doc.size()[0])
            top_pairwise_scores = top_pairwise_slow_scores + top_pairwise_fast_scores
        else:
            top_pairwise_fast_scores, top_antecedent_idx = torch.topk(pairwise_fast_scores, k=pairwise_fast_scores.size(0))
            top_antecedent_mask = batch_select(antecedent_mask, top_antecedent_idx, device)  # [num top spans, max top antecedents]
            top_antecedent_offsets = batch_select(antecedent_offsets, top_antecedent_idx, device)
            
            top_pairwise_scores = top_pairwise_fast_scores  # [num top spans, max top antecedents]
        
        top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)
        
            
        if not do_loss:
            if type(self.mention_proposer) == CFGMentionProposer or self.config["mention_sigmoid"]:
                top_antecedent_log_p_mention = top_span_log_p_mention[top_antecedent_idx]
                log_norm = logsumexp(top_antecedent_scores, dim=1)
                # Shape: (num_spans_to_keep, max_antecedents+1)
                log_p_im = top_antecedent_scores - log_norm.unsqueeze(-1) + top_span_log_p_mention.unsqueeze(-1) 
                # Shape: (num_spans_to_keep)
                log_p_em = torch.logaddexp(
                    log1mexp(top_span_log_p_mention),
                    top_span_log_p_mention - log_norm + torch.finfo(log_norm.dtype).eps
                )
                # log probability for inference
                log_probs = torch.cat([log_p_em.unsqueeze(-1), log_p_im[:,1:]], dim=-1)
                
                return candidate_starts, candidate_ends, candidate_mention_parsing_scores, top_span_starts, top_span_ends, top_antecedent_idx, log_probs
            elif type(self.mention_proposer) == GreedyMentionProposer:
                return candidate_starts, candidate_ends, candidate_mention_parsing_scores, top_span_starts, top_span_ends, top_antecedent_idx, top_antecedent_scores
        
        log_norm = logsumexp(top_antecedent_scores, dim=1)
        if type(self.mention_proposer) == CFGMentionProposer or self.config["mention_sigmoid"]:
            top_antecedent_log_p_mention = top_span_log_p_mention[top_antecedent_idx] 
            # Shape: (num_spans_to_keep, max_antecedents+1)
            log_p_im = top_antecedent_scores - log_norm.unsqueeze(-1) + top_span_log_p_mention.unsqueeze(-1)
            # Shape: (num_spans_to_keep)
            log_p_em = torch.logaddexp(
                log1mexp(top_span_log_p_mention) + torch.finfo(log_norm.dtype).eps, 
                top_span_log_p_mention - log_norm + torch.finfo(log_norm.dtype).eps
            )
            # log probability for inference
            log_probs = torch.cat([log_p_em.unsqueeze(-1), log_p_im[:,1:]], dim=-1)

        # Get gold labels
        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedent_idx]
        top_antecedent_cluster_ids += (top_antecedent_mask.long() - 1) * 100000  # Mask id on invalid antecedents
        same_gold_cluster_indicator = (top_antecedent_cluster_ids == torch.unsqueeze(top_span_cluster_ids, 1))
        non_dummy_indicator = non_dummy_indicator[selected_idx] # (top_span_cluster_ids > 0).squeeze()
        # non_dummy_indicator is the coreferable flags
        pairwise_labels = same_gold_cluster_indicator & torch.unsqueeze(top_span_cluster_ids > 0, 1)
        dummy_antecedent_labels = torch.logical_not(pairwise_labels.any(dim=1, keepdims=True))
        
        top_antecedent_gold_labels = torch.cat([dummy_antecedent_labels, pairwise_labels], dim=1)
        
        # Get loss
        if type(self.mention_proposer) == CFGMentionProposer or self.config["mention_sigmoid"]:
            coref_loss = -logsumexp(log_p_im + top_antecedent_gold_labels.log(), dim=-1) # for mentions
            loss = mp_loss + (coref_loss * non_dummy_indicator).sum() + (-log_p_em * torch.logical_not(non_dummy_indicator)).sum()
            return [candidate_starts, candidate_ends, candidate_mention_parsing_scores, top_span_starts, top_span_ends, top_antecedent_idx, log_probs], loss
        elif type(self.mention_proposer) == GreedyMentionProposer:
            log_marginalized_antecedent_scores = logsumexp(top_antecedent_scores + top_antecedent_gold_labels.log(), dim=1)
            loss = (log_norm - log_marginalized_antecedent_scores).sum()

            return [candidate_starts, candidate_ends, candidate_mention_parsing_scores, top_span_starts, top_span_ends, top_antecedent_idx, top_antecedent_scores], loss
            
    def _extract_top_spans(self, candidate_idx_sorted, candidate_starts, candidate_ends, num_top_spans):
        """ Keep top non-cross-overlapping candidates ordered by scores; compute on CPU because of loop """
        selected_candidate_idx = []
        start_to_max_end, end_to_min_start = {}, {}
        for candidate_idx in candidate_idx_sorted:
            if len(selected_candidate_idx) >= num_top_spans:
                break
            # Perform overlapping check
            span_start_idx = candidate_starts[candidate_idx]
            span_end_idx = candidate_ends[candidate_idx]
            cross_overlap = False
            for token_idx in range(span_start_idx, span_end_idx + 1):
                max_end = start_to_max_end.get(token_idx, -1)
                if token_idx > span_start_idx and max_end > span_end_idx:
                    cross_overlap = True
                    break
                min_start = end_to_min_start.get(token_idx, -1)
                if token_idx < span_end_idx and 0 <= min_start < span_start_idx:
                    cross_overlap = True
                    break
            if not cross_overlap:
                # Pass check; select idx and update dict stats
                selected_candidate_idx.append(candidate_idx)
                max_end = start_to_max_end.get(span_start_idx, -1)
                if span_end_idx > max_end:
                    start_to_max_end[span_start_idx] = span_end_idx
                min_start = end_to_min_start.get(span_end_idx, -1)
                if min_start == -1 or span_start_idx < min_start:
                    end_to_min_start[span_end_idx] = span_start_idx
        # Sort selected candidates by span idx
        selected_candidate_idx = sorted(selected_candidate_idx, key=lambda idx: (candidate_starts[idx], candidate_ends[idx]))
        if len(selected_candidate_idx) < num_top_spans:  # Padding
            selected_candidate_idx += ([selected_candidate_idx[0]] * (num_top_spans - len(selected_candidate_idx)))
        return selected_candidate_idx

    def get_predicted_antecedents(self, antecedent_idx, antecedent_scores):
        """ CPU list input """
        predicted_antecedents = []
        for i, idx in enumerate((antecedent_scores.argmax(dim=1) - 1).tolist()):
            if idx < 0:
                predicted_antecedents.append(-1)
            elif idx >= len(antecedent_idx[0]):
                predicted_antecedents.append(-2)
            else:
                predicted_antecedents.append(antecedent_idx[i][idx])
        return predicted_antecedents

    def get_predicted_clusters(self, span_starts, span_ends, antecedent_idx, antecedent_scores):
        """ CPU list input """
        # Get predicted antecedents
        predicted_antecedents = self.get_predicted_antecedents(antecedent_idx, antecedent_scores)

        # Get predicted clusters
        mention_to_cluster_id = {}
        predicted_clusters = []
        for i, predicted_idx in enumerate(predicted_antecedents):
            if predicted_idx == -1:
                continue
            elif predicted_idx == -2:
                cluster_id = len(predicted_clusters)
                predicted_clusters.append([(int(span_starts[i]), int(span_ends[i]))])
                mention_to_cluster_id[(int(span_starts[i]), int(span_ends[i]))] = cluster_id
                continue
            assert i > predicted_idx, f'span idx: {i}; antecedent idx: {predicted_idx}'
            # Check antecedent's cluster
            antecedent = (int(span_starts[predicted_idx]), int(span_ends[predicted_idx]))
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1)
            
            if antecedent_cluster_id == -1:
                antecedent_cluster_id = len(predicted_clusters)
                predicted_clusters.append([antecedent])
                mention_to_cluster_id[antecedent] = antecedent_cluster_id
                
            # Add mention to cluster
            mention = (int(span_starts[i]), int(span_ends[i]))
            predicted_clusters[antecedent_cluster_id].append(mention)
            mention_to_cluster_id[mention] = antecedent_cluster_id

        predicted_clusters = [tuple(c) for c in predicted_clusters]
        return predicted_clusters, mention_to_cluster_id, predicted_antecedents

    def update_evaluator(self, span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator):
        predicted_clusters, mention_to_cluster_id, _ = self.get_predicted_clusters(span_starts, span_ends, antecedent_idx, antecedent_scores)
        mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
        gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        
        # gold mentions
        gms = set([x for cluster in gold_clusters for x in cluster])
        # getting meta informations, e.g. nested depth, width
        metainfo_gms = defaultdict(lambda: defaultdict(int))
        for x in gms:
            metainfo_gms[x]["width"] = x[1] - x[0]
            for y in gms:
                if y[0] <= x[0] and y[1] >= x[1]:
                    metainfo_gms[x]["depth"] += 1
                
        
        recalled_gms = set([(int(x), int(y)) for x,y in zip(span_starts, span_ends)])
        self.all_pred_men += len(recalled_gms)
        
        evaluator.update(
            predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold, 
            metainfo_gms, recalled_gms
        )
        return predicted_clusters

