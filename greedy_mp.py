import logging
import math
from typing import Any, Dict, List, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

logger = logging.getLogger(__name__)

class GreedyMentionProposer(torch.nn.Module):
    
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        

    def forward(
        self,
        spans: torch.IntTensor,
        span_mention_scores: torch.FloatTensor,
        span_mask: torch.FloatTensor,
        token_num: torch.IntTensor,
        num_spans_to_keep: int,
        take_top_spans_per_sentence = False,
        flat_span_sent_ids = None,
        ratio = 0.4,
    ):  
        if not take_top_spans_per_sentence:
            top_span_indices = masked_topk_non_overlap(
                span_mention_scores,
                span_mask, 
                num_spans_to_keep,
                spans
            )
            top_spans = spans[top_span_indices]
            top_span_scores = span_mention_scores[top_span_indices]
            return top_span_scores, top_span_indices, top_spans, 0., None
        else:
            top_span_indices, top_span_scores, top_spans = [], [], []
            prev_sent_id, prev_span_id = 0, 0
            for span_id, sent_id in enumerate(flat_span_sent_ids.tolist()):
                if sent_id != prev_sent_id:
                    sent_span_indices = masked_topk_non_overlap(
                        span_mention_scores[prev_span_id:span_id],
                        span_mask[prev_span_id:span_id], 
                        int(ratio * (token_num[prev_sent_id])), # [CLS], [SEP]
                        spans[prev_span_id:span_id],
                        non_crossing=True,
                    ) + prev_span_id
                    
                    top_span_indices.append(sent_span_indices)
                    top_span_scores.append(span_mention_scores[sent_span_indices])
                    top_spans.append(spans[sent_span_indices])
                    
                    prev_sent_id, prev_span_id = sent_id, span_id
            # last sentence
            sent_span_indices = masked_topk_non_overlap(
                        span_mention_scores[prev_span_id:],
                        span_mask[prev_span_id:], 
                        int(ratio * (token_num[-1])),
                        spans[prev_span_id:],
                        non_crossing=True,
                    ) + prev_span_id
            
            top_span_indices.append(sent_span_indices)
            top_span_scores.append(span_mention_scores[sent_span_indices])
            top_spans.append(spans[sent_span_indices])
            
            num_top_spans = [x.size(0) for x in top_span_indices]
            max_num_top_span = max(num_top_spans)
            
            top_spans = torch.stack(
                [torch.cat([x, x.new_zeros((max_num_top_span-x.size(0), 2))], dim=0) for x in top_spans], dim=0
            )
            top_span_masks = torch.stack(
                [torch.cat([x.new_ones((x.size(0), )), x.new_zeros((max_num_top_span-x.size(0), ))], dim=0) for x in top_span_indices], dim=0
            )
            top_span_indices = torch.stack(
                [torch.cat([x, x.new_zeros((max_num_top_span-x.size(0), ))], dim=0) for x in top_span_indices], dim=0
            )
            
            return top_span_scores, top_span_indices, top_spans, 0., None, top_span_masks
        

def masked_topk_non_overlap(
        span_scores,
        span_mask, 
        num_spans_to_keep,
        spans,
        non_crossing=True
    ):
    
    sorted_scores, sorted_indices = torch.sort(span_scores + span_mask.log(), descending=True)
    sorted_indices = sorted_indices.tolist()
    spans = spans.tolist()
    
    if not non_crossing:
        selected_candidate_idx = sorted(sorted_indices[:num_spans_to_keep], key=lambda idx: (spans[idx][0], spans[idx][1]))
        selected_candidate_idx = span_scores.new_tensor(selected_candidate_idx, dtype=torch.long)
        return selected_candidate_idx
    
    selected_candidate_idx = []
    start_to_max_end, end_to_min_start = {}, {}
    for candidate_idx in sorted_indices:
        if len(selected_candidate_idx) >= num_spans_to_keep:
            break
        # Perform overlapping check
        span_start_idx = spans[candidate_idx][0]
        span_end_idx = spans[candidate_idx][1]
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
    selected_candidate_idx = sorted(selected_candidate_idx, key=lambda idx: (spans[idx][0], spans[idx][1]))
    selected_candidate_idx = span_scores.new_tensor(selected_candidate_idx, dtype=torch.long)
    
    return selected_candidate_idx
