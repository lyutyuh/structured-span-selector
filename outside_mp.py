import logging
import math
from typing import Any, Dict, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

from util import logsumexp, clip_to_01, stripe, masked_topk_non_overlap

from genbmm import logbmm, logbmminside

logger = logging.getLogger(__name__)
LARGENUMBER = 1e4

class CKY(torch.nn.Module):
    def __init__(
        self,
        max_span_width=30,
    ):
        super().__init__()
        self.max_span_width = max_span_width
    
    def forward(
        self,
        span_mention_score_matrix: torch.FloatTensor, 
        sequence_lengths: torch.IntTensor,
   ) -> Tuple[torch.FloatTensor]:
        
        with torch.autograd.enable_grad():
            # Enable gradients during inference
            return self.coolio(span_mention_score_matrix, sequence_lengths)
          
    def coolio(
        self, 
        span_mention_score_matrix: torch.FloatTensor, 
        sequence_lengths: torch.IntTensor,
    ) -> Tuple[torch.FloatTensor]:
        """
            Parameters:
                span_mention_score_matrix: shape (batch_size, sent_len, sent_len, score_dim)
                    Score of each span being a span of interest. There are batch_size number
                    of sentences in this document. And the maximum length of sentence is 
                    sent_len. 
                sequence_lengths: shape (batch_size, )
                    The actual length of each sentence. 
        """
        # faster inside-outside
        
        span_mention_score_matrix.requires_grad_(True)
        
        batch_size, _, _, score_dim = span_mention_score_matrix.size()
        seq_len = sequence_lengths.max()
        # Shape: (batch_size, )
        sequence_lengths = sequence_lengths.view(-1)
        
        rules = span_mention_score_matrix
        log1p_exp_rules = torch.log1p(rules.squeeze(-1).exp())
        
        zero_rules = (rules.new_ones(seq_len, seq_len).tril(diagonal=-1))*(-LARGENUMBER)
        zero_rules = zero_rules.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1,1,1)
        
        inside_s = torch.cat([rules.clone(), zero_rules], dim=3)
        inside_s = inside_s.logsumexp(dim=3)
        
        diag_mask = torch.zeros_like(inside_s[0])
            
        for width in range(0, seq_len-1):
            inside_s = logbmminside(inside_s, width+1)
            inside_s = inside_s + torch.diagonal_scatter(
                diag_mask, torch.ones_like(diag_mask.diagonal(width+1)), width+1
            ).unsqueeze(0) * log1p_exp_rules
            
        series_batchsize = torch.arange(0, batch_size, dtype=torch.long)
        Z = inside_s[series_batchsize, 0, sequence_lengths-1] # (batch_size, )
        
        marginal = torch.autograd.grad(
            Z.sum(),
            span_mention_score_matrix,
            create_graph=True,
            only_inputs=True,
            allow_unused=False,
        )
        marginal = marginal[0].squeeze()
        return (Z.view(-1), marginal)  # Shape: (batch_size, seq_len, seq_len, )
        
    def io(
        self, 
        span_mention_score_matrix: torch.FloatTensor, 
        sequence_lengths: torch.IntTensor,
    ) -> Tuple[torch.FloatTensor]:
        """
            Parameters:
                span_mention_score_matrix: shape (batch_size, sent_len, sent_len, score_dim)
                    Score of each span being a span of interest. There are batch_size number
                    of sentences in this document. And the maximum length of sentence is 
                    sent_len. 
                sequence_lengths: shape (batch_size, )
                    The actual length of each sentence. 
        """
        # inside-outside
        
        span_mention_score_matrix.requires_grad_(True)
        
        batch_size, _, _, score_dim = span_mention_score_matrix.size()
        seq_len = sequence_lengths.max()
        # Shape: (batch_size, )
        sequence_lengths = sequence_lengths.view(-1)
        
        # Shape: (seq_len, seq_len, score_dim, batch_size)
        span_mention_score_matrix = span_mention_score_matrix.permute(1, 2, 3, 0)
        
        # There should be another matrix of non-mention span scores, which is full of 0s
        # Shape: (seq_len, seq_len, score_dim + 1, batch_size), 2 for mention / non-mention
        inside_s = span_mention_score_matrix.new_zeros(seq_len, seq_len, score_dim + 1, batch_size)

        for width in range(0, seq_len):
            n = seq_len - width
            if width == 0:
                inside_s[:,:,:score_dim,:].diagonal(width).copy_(
                    span_mention_score_matrix.diagonal(width)
                )
                continue

            # [n, width, score_dim + 1, batch_size]
            split_1 = stripe(inside_s, n, width)
            split_2 = stripe(inside_s, n, width, (1, width), 0)

            # [n, width, batch_size]
            inside_s_span = logsumexp(split_1, 2) + logsumexp(split_2, 2)
            # [1, batch_size, n]
            inside_s_span = logsumexp(inside_s_span, 1, keepdim=True).permute(1, 2, 0)

            if width < self.max_span_width:
                inside_s.diagonal(width).copy_(
                    torch.cat(
                        [inside_s_span + span_mention_score_matrix.diagonal(width), # mention
                         inside_s_span],                                            # non-mention
                    dim=0
                    )
                )
            else:
                inside_s.diagonal(width).copy_(
                    torch.cat(
                        [torch.full_like(span_mention_score_matrix.diagonal(width), -LARGENUMBER),  # mention
                         inside_s_span],                                                            # non-mention
                    dim=0
                    )
                )

        inside_s = inside_s.permute(0,1,3,2) # (seq_len, seq_len, batch_size, 2), 2 for mention / non-mention
        series_batchsize = torch.arange(0, batch_size, dtype=torch.long)
        
        Z = logsumexp(inside_s[0, sequence_lengths-1, series_batchsize], dim=-1) # (batch_size,)
        
        marginal = torch.autograd.grad(
            Z.sum(),
            span_mention_score_matrix,
            create_graph=True,
            only_inputs=True,
            allow_unused=False,
        )
        marginal = marginal[0].squeeze()
        return (Z.view(-1), marginal.permute(2,0,1)) # Shape: (batch_size, seq_len, seq_len, ) 
    
    @staticmethod
    def viterbi(
        span_mention_score_matrix: torch.FloatTensor, 
        sequence_lengths: torch.IntTensor,
       ) -> Tuple[torch.FloatTensor]:
        
        if len(span_mention_score_matrix.size()) == 4:
            span_mention_score_matrix, _ = span_mention_score_matrix.max(-1)
        # Shape: (seq_len, seq_len, batch_size)
        span_mention_score_matrix = span_mention_score_matrix.permute(1, 2, 0)
        
        
        # Shape: (batch_size, )
        sequence_lengths = sequence_lengths.view(-1)
        # There should be another matrix of non-mention span scores, which is full of 0s
        seq_len, _, batch_size = span_mention_score_matrix.size()
        
        s = span_mention_score_matrix.new_zeros(seq_len, seq_len, 2, batch_size)
        p = sequence_lengths.new_zeros(seq_len, seq_len, 2, batch_size) # backtrack
        
        for width in range(0, seq_len):
            n = seq_len - width
            span_score = span_mention_score_matrix.diagonal(width)
            if width == 0:
                s.diagonal(0)[0, :].copy_(span_score)
                continue
            # [n, width, 2, 1, batch_size]
            split1 = stripe(s, n, width, ).unsqueeze(3)
            # [n, width, 1, 2, batch_size]
            split2 = stripe(s, n, width, (1, width), 0).unsqueeze(2)
            
            # [n, width, 2, 2, batch_size]
            s_span = split1 + split2
            # [batch_size, n, 2, width, 2, 2]
            s_span = s_span.permute(4, 0, 1, 2, 3).unsqueeze(2).repeat(1,1,2,1,1,1)
            
            s_span[:,:,0,:,:,:] += span_score.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
             # [batch_size, n, 2]
            s_span, p_span = s_span.view(batch_size, n, 2, -1).max(-1) # best split
            s.diagonal(width).copy_(
                s_span.permute(2, 0, 1)
            )
            starts = p.new_tensor(range(n)).unsqueeze(0).unsqueeze(0)
            p.diagonal(width).copy_(
                p_span.permute(2, 0, 1) + starts * 4
            )
            
        def backtrack(mat, p, i, j, c):
            if j == i:
                return p.new_tensor([(i, j, c)], dtype=torch.long)
            int_pijc = int(p[i][j][c])
            split = int_pijc // 4
            ltree = backtrack(mat, p, i, split, (int_pijc // 2) % 2)
            rtree = backtrack(mat, p, split+1, j, int_pijc % 2)
            return torch.cat([p.new_tensor([(i,j,c)], dtype=torch.long),  ltree, rtree], dim=0)

        # (batch_size, seq_len, seq_len, 2)
        p = p.permute(3, 0, 1, 2)
        
        span_mention_score_matrix_cpu = span_mention_score_matrix.cpu()
        p_cpu = p.cpu()
        
        trees = [backtrack(s[:,:,:,i], p_cpu[i], 0, int(sequence_lengths[i]-1), 
                           (int) (s[0,int(sequence_lengths[i]-1),0,i] < s[0,int(sequence_lengths[i]-1),1,i]))
                 for i in range(batch_size)]
        
        return trees 
    
    def outside(
            self,
            inside_s: torch.FloatTensor,
            span_mention_score_matrix: torch.FloatTensor,
            sequence_lengths: torch.IntTensor,
           ):
        '''
        inside_s: Shape: (seq_len, seq_len, 2, batch_size), 2 for mention / non-mention
        span_mention_score_matrix: Shape: (seq_len, seq_len, batch_size)

        Return: outside_s : Shape: (seq_len, seq_len, 2, batch_size), 2 for mention / non-mention
        '''
        seq_len = sequence_lengths.max()
        
        _, _, batch_size = span_mention_score_matrix.size()
        series_batchsize = torch.arange(0, batch_size, dtype=torch.long)
        
        # Shape: (seq_len, seq_len, batch_size)
        # outside_s = span_mention_score_matrix.new_zeros(seq_len, seq_len, batch_size)
        outside_s = span_mention_score_matrix.new_full((seq_len, seq_len, batch_size), fill_value=-LARGENUMBER)

        mask_top = span_mention_score_matrix.new_zeros(seq_len, seq_len, batch_size).bool()
        mask_top[0, sequence_lengths-1, series_batchsize] = 1

        for width in range(seq_len-1, -1, -1):
            n = seq_len - width
            if width == seq_len-1:
                continue
            outside_s[mask_top] = 0
            # [n, n, 1,  2, batch_size]
            split_1 = inside_s[:n-1,:n-1].unsqueeze(2) # using the upper triangular [:n, :n] of the inside matrix
            # [n, n, 2,  1, batch_size]
            split_2 = outside_s[:n-1, width+1:seq_len].unsqueeze(2).unsqueeze(3).repeat(1,1,2,1,1) # using a submatrix of the outside matrix [:n, width:seq_len]
            # [n, n, 1, batch_size]
            span_score_submatrix = span_mention_score_matrix[:n-1, width+1:seq_len].unsqueeze(2)
            # [n, n, 1,  2, batch_size]
            split_3 = inside_s[width+1:seq_len, width+1:seq_len].unsqueeze(2) # using the upper triangular [width:seq_len, width:seq_len] of the inside matrix

            # [n, n, 1,  1, 1]
            upp_triu_mask = torch.triu(span_score_submatrix.new_ones(n-1,n-1), diagonal=0).view(n-1,n-1,1,1,1)

            # [n, n, 2,  2,  batch_size] 
            # B -> CA,   B, C, A \in {0,1}
            outside_s_span_1 = (split_1 + split_2)
            outside_s_span_1[:,:,0,:,:] += span_score_submatrix

            outside_s_span_1 += (upp_triu_mask*LARGENUMBER - LARGENUMBER) #  upp_triu_mask.log() # 
            # [batch_size, n, n, 2,  2]
            outside_s_span_1 = outside_s_span_1.permute(4, 1, 0, 2, 3)
            # [batch_size, n]
            outside_s_span_1 = logsumexp(outside_s_span_1.reshape(batch_size, n-1, -1), dim=-1)
            # outside_s_span_1.logsumexp((1,3,4)) # sum vertical, as right child

            # [n, n, 2,  2,  batch_size] 
            outside_s_span_2 = (split_3 + split_2)
            outside_s_span_2[:,:,0,:,:] += span_score_submatrix

            outside_s_span_2 +=  (upp_triu_mask*LARGENUMBER - LARGENUMBER) # upp_triu_mask.log() #
            # [batch_size, n, n, 2,  2]
            outside_s_span_2 = outside_s_span_2.permute(4, 0, 1, 2, 3)
            # [batch_size, n]
            outside_s_span_2 = logsumexp(outside_s_span_2.view(batch_size, n-1, -1), dim=-1) # sum horizontal, as left child

            # shift and sum
            outside_s_span_1 = torch.cat([outside_s_span_1.new_tensor([float(-LARGENUMBER)]*batch_size).unsqueeze(-1), outside_s_span_1], dim=-1)
            outside_s_span_2 = torch.cat([outside_s_span_2, outside_s_span_2.new_tensor([float(-LARGENUMBER)]*batch_size).unsqueeze(-1)], dim=-1)

            # [batch_size, n, 2]
            outside_s_span = torch.stack([outside_s_span_1, outside_s_span_2], dim=-1)

            # [batch_size, n]
            outside_s_span = logsumexp(outside_s_span, dim=-1)

            outside_s.diagonal(width).copy_(outside_s_span)
            
        return outside_s


def get_sentence_matrix(
    sentence_num, 
    max_sentence_length, 
    unidimensional_values,
    span_location_indices,
    padding_value=0.
):
    total_units = sentence_num * max_sentence_length * max_sentence_length 
    flat_matrix_by_sentence = unidimensional_values.new_full(
        (total_units, unidimensional_values.size(-1)), padding_value
    ).index_copy(0, span_location_indices, unidimensional_values.view(-1, unidimensional_values.size(-1)))
    
    return flat_matrix_by_sentence.view(sentence_num, max_sentence_length, max_sentence_length, unidimensional_values.size(-1))
    

class CFGMentionProposer(torch.nn.Module):
    def __init__(
        self,
        max_span_width=30,
        neg_sample_rate=0.2,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.neg_sample_rate = float(neg_sample_rate)
        self.cky_module = CKY(max_span_width)

    def forward(
        self,  # type: ignore
        spans: torch.IntTensor,
        span_mention_scores: torch.FloatTensor,
        span_mask: torch.FloatTensor,
        span_labels: torch.IntTensor,
        sentence_lengths: torch.IntTensor,
        num_spans_to_keep: int,
        flat_span_location_indices: torch.IntTensor,
        take_top_spans_per_sentence = False,
        flat_span_sent_ids = None,
        ratio = 0.
    ):
        # Shape: (batch_size, document_length, embedding_size)
        num_spans = spans.size(1)
        span_max_item = spans.max()
        
        sentence_offsets = torch.cumsum(sentence_lengths.squeeze(), 0)
        sentence_offsets = torch.cat(
            [sentence_offsets.new_zeros(1, 1), sentence_offsets.view(1, -1)], 
            dim=-1
        )
        span_mention_scores = span_mention_scores + (span_mask.unsqueeze(-1) * LARGENUMBER - LARGENUMBER)
        max_sentence_length = sentence_lengths.max()
        sentence_num = sentence_lengths.size(0)
        
        # We directly calculate indices of span scores in the matrices during data preparation.
        # The indices is 1-d (except the batch dimension) to facilitate index_copy_
        # We copy the span scores into (batch_size, sentence_num, max_sentence_length, max_sentence_length)
        # shaped score matrices
        
        # We will do sentence-level CKY over span scores
        # span_mention_scores shape: (batch_size, num_spans, 2)
        # the first column scores are for parsing, the second column for linking
        
        span_score_matrix_by_sentence = get_sentence_matrix(
            sentence_num, max_sentence_length, span_mention_scores,
            flat_span_location_indices, padding_value=-LARGENUMBER
        ) 
        valid_span_flag_matrix_by_sentence = get_sentence_matrix(
            sentence_num, max_sentence_length, torch.ones_like(span_mask).unsqueeze(-1), 
            flat_span_location_indices, padding_value=0
        ).squeeze(-1)
        
        
        Z, marginal = self.cky_module(
            span_score_matrix_by_sentence, sentence_lengths
        )
        span_marginal = torch.masked_select(marginal, valid_span_flag_matrix_by_sentence)
        
        if not take_top_spans_per_sentence:
            top_span_indices = masked_topk_non_overlap(
                span_marginal,
                span_mask, 
                num_spans_to_keep,
                spans
            )
            span_marginal = clip_to_01(span_marginal)
        
            top_ind_list = top_span_indices.tolist()
            all_ind_list = list(range(0, span_marginal.size(0)))
            neg_sample_indices = np.random.choice(
                list(set(all_ind_list) - set(top_ind_list)), 
                int(self.neg_sample_rate * num_spans_to_keep), 
                False
            )
            neg_sample_indices = top_span_indices.new_tensor(sorted(neg_sample_indices))
        else:
            top_span_indices, sentwise_top_span_marginal, top_spans = [], [], []
            prev_sent_id, prev_span_id = 0, 0
            for span_id, sent_id in enumerate(flat_span_sent_ids.tolist()):
                if sent_id != prev_sent_id:
                    sent_span_indices = masked_topk_non_overlap(
                        span_marginal[prev_span_id:span_id],
                        span_mask[prev_span_id:span_id], 
                        int(ratio * sentence_lengths[prev_sent_id]),
                        spans[prev_span_id:span_id],
                        non_crossing=False,
                    ) + prev_span_id
                    
                    top_span_indices.append(sent_span_indices)
                    sentwise_top_span_marginal.append(span_marginal[sent_span_indices])
                    top_spans.append(spans[sent_span_indices])
                    
                    prev_sent_id, prev_span_id = sent_id, span_id
            # last sentence
            sent_span_indices = masked_topk_non_overlap(
                        span_marginal[prev_span_id:],
                        span_mask[prev_span_id:], 
                        int(ratio * sentence_lengths[-1]),
                        spans[prev_span_id:],
                        non_crossing=False,
                    ) + prev_span_id
            
            top_span_indices.append(sent_span_indices)
            sentwise_top_span_marginal.append(span_marginal[sent_span_indices])
            top_spans.append(spans[sent_span_indices])
            
            num_top_spans = [x.size(0) for x in top_span_indices]
            max_num_top_span = max(num_top_spans)
            
            sentwise_top_span_marginal = torch.stack(
                [torch.cat([x, x.new_zeros((max_num_top_span-x.size(0), ))], dim=0) for x in sentwise_top_span_marginal], dim=0
            )
            top_spans = torch.stack(
                [torch.cat([x, x.new_zeros((max_num_top_span-x.size(0), 2))], dim=0) for x in top_spans], dim=0
            )
            top_span_masks = torch.stack(
                [torch.cat([spans.new_ones((x, )), spans.new_zeros((max_num_top_span-x, ))], dim=0) for x in num_top_spans], dim=0
            )
            
            flat_top_span_indices = torch.cat(top_span_indices, dim=0)
            top_span_indices = torch.stack(
                [torch.cat([x, x.new_zeros((max_num_top_span-x.size(0), ))], dim=0) for x in top_span_indices], dim=0
            )
            
            top_ind_list = flat_top_span_indices.tolist()
            all_ind_list = list(range(0, span_marginal.size(0)))
            neg_sample_indices = np.random.choice(
                list(set(all_ind_list) - set(top_ind_list)), 
                int(self.neg_sample_rate * num_spans_to_keep), False
            )
            neg_sample_indices = top_span_indices.new_tensor(sorted(neg_sample_indices))
            pass # End else
        
        
        if not self.training:
            with torch.no_grad():
                best_trees = CKY.viterbi(span_score_matrix_by_sentence.detach(), sentence_lengths)
                
            best_tree_spans = [(x[:,:2]+offset).cuda() for x, offset in zip(best_trees, sentence_offsets.view(-1).cpu())]
            
            best_tree_tags = torch.cat([x[:,-1] for x in best_trees], dim=-1).cuda()
            best_tree_spans = torch.cat(best_tree_spans, dim=0).cuda()
            best_tree_span_mask = (best_tree_tags == 0).unsqueeze(-1)
            if best_tree_span_mask.sum() > 0:
                # top spans per sentence
                helper_matrix = span_mask.new_zeros(span_max_item+1, span_max_item+1)
                top_spans = torch.masked_select(best_tree_spans, best_tree_span_mask).view(-1, 2)
                helper_matrix[top_spans[:,0],top_spans[:,1]] |= torch.tensor(True)
                top_span_mask = helper_matrix[spans[:,0],spans[:,1]]
                top_span_indices = torch.nonzero(top_span_mask, as_tuple=True)[0]
                
                if take_top_spans_per_sentence:
                    sentwise_top_span_indices, sentwise_top_span_marginal, sentwise_top_spans = [], [], []
                    prev_sent_id, prev_span_id = 0, 0
                    
                    for span_id, sent_id in enumerate(flat_span_sent_ids.tolist()):
                        if sent_id != prev_sent_id:
                            current_sentence_indices = torch.nonzero(top_span_mask[prev_span_id:span_id], as_tuple=True)[0] # unshifted
                            sentwise_top_span_indices.append(current_sentence_indices + prev_span_id)
                            sentwise_top_span_marginal.append(span_marginal[prev_span_id:span_id][current_sentence_indices])
                            sentwise_top_spans.append(spans[prev_span_id:span_id][current_sentence_indices])
                    
                            prev_sent_id, prev_span_id = sent_id, span_id
                    
                    current_sentence_indices = torch.nonzero(top_span_mask[prev_span_id:], as_tuple=True)[0] # unshifted
                    sentwise_top_span_indices.append(current_sentence_indices + prev_span_id)
                    sentwise_top_span_marginal.append(span_marginal[prev_span_id:][current_sentence_indices])
                    sentwise_top_spans.append(spans[prev_span_id:][current_sentence_indices])
                    
                    num_top_spans = [x.size(0) for x in sentwise_top_span_indices]
                    max_num_top_span = max(num_top_spans)
                    
                    top_spans = torch.stack(
                        [torch.cat([x, x.new_zeros((max_num_top_span-x.size(0), 2))], dim=0) for x in sentwise_top_spans], dim=0
                    )
                    top_span_masks = torch.stack(
                        [torch.cat([spans.new_ones((x, )), spans.new_zeros((max_num_top_span-x, ))], dim=0) for x in num_top_spans], dim=0
                    )
                    top_span_indices = torch.stack(
                        [torch.cat([x, x.new_zeros((max_num_top_span-x.size(0), ))], dim=0) for x in sentwise_top_span_indices], dim=0
                    )
                    sentwise_top_span_marginal = torch.stack(
                        [torch.cat([x, x.new_zeros((max_num_top_span-x.size(0), ))], dim=0) for x in sentwise_top_span_marginal], dim=0
                    )
            else:
                logger.info("expected %d but %d in CKY parse, not using CKY parse" % (num_spans_to_keep, int(best_tree_span_mask.sum())))
                pass
            
        
        if self.training and neg_sample_indices.size(0) > 0:
            if take_top_spans_per_sentence:
                not_mention_loss = -(1 - span_marginal[neg_sample_indices]).log()
                loss = not_mention_loss.mean() * (self.neg_sample_rate * num_spans_to_keep)
            else:
                non_mention_flag = (span_labels <= 0)
                # -log(1 - P(m))
                not_mention_loss = -(1 - span_marginal[neg_sample_indices]).log() * non_mention_flag[neg_sample_indices]
                loss = not_mention_loss.mean() * (self.neg_sample_rate * num_spans_to_keep)
        else:
            loss = 0.
            
        if not take_top_spans_per_sentence:
            top_spans = spans[top_span_indices]
            return span_marginal, top_span_indices, top_spans, loss, None
        else:
            
            return sentwise_top_span_marginal, top_span_indices, top_spans, loss, None, top_span_masks

