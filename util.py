from os import makedirs
from os.path import join
import numpy as np
import pyhocon
import logging
import torch
import random

logger = logging.getLogger(__name__)


def flatten(l):
    return [item for sublist in l for item in sublist]


def initialize_config(config_name, task="coref"):
    logger.info("Running {} experiment: {}".format(task, config_name))

    if task == "coref":
        config = pyhocon.ConfigFactory.parse_file("experiments.conf")[config_name]
    elif task == "srl":
        config = pyhocon.ConfigFactory.parse_file("experiments_srl.conf")[config_name]
        
    config['log_dir'] = join(config["log_root"], config_name)
    makedirs(config['log_dir'], exist_ok=True)

    config['tb_dir'] = join(config['log_root'], 'tensorboard')
    makedirs(config['tb_dir'], exist_ok=True)

    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def set_seed(seed, set_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if set_gpu and torch.cuda.is_available():
        # Necessary for reproducibility; lower performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
    logger.info('Random seed is set to %d' % seed)


def bucket_distance(offsets):
    """ offsets: [num spans1, num spans2] """
    # 10 semi-logscale bin: 0, 1, 2, 3, 4, (5-7)->5, (8-15)->6, (16-31)->7, (32-63)->8, (64+)->9
    logspace_distance = torch.log2(offsets.to(torch.float)).to(torch.long) + 3
    identity_mask = (offsets <= 4).to(torch.long)
    combined_distance = identity_mask * offsets + (1 - identity_mask) * logspace_distance
    combined_distance = torch.clamp(combined_distance, 0, 9)
    return combined_distance


def batch_select(tensor, idx, device=torch.device('cpu')):
    """ Do selection per row (first axis). """
    assert tensor.shape[0] == idx.shape[0]  # Same size of first dim
    dim0_size, dim1_size = tensor.shape[0], tensor.shape[1]

    tensor = torch.reshape(tensor, [dim0_size * dim1_size, -1])
    idx_offset = torch.unsqueeze(torch.arange(0, dim0_size, device=device) * dim1_size, 1)
    new_idx = idx + idx_offset
    selected = tensor[new_idx]

    if tensor.shape[-1] == 1:  # If selected element is scalar, restore original dim
        selected = torch.squeeze(selected, -1)

    return selected


def batch_add(tensor, idx, val, device=torch.device('cpu')):
    """ Do addition per row (first axis). """
    assert tensor.shape[0] == idx.shape[0]  # Same size of first dim
    dim0_size, dim1_size = tensor.shape[0], tensor.shape[1]

    tensor = torch.reshape(tensor, [dim0_size * dim1_size, -1])
    idx_offset = torch.unsqueeze(torch.arange(0, dim0_size, device=device) * dim1_size, 1)
    new_idx = idx + idx_offset
    
    val = val.reshape(val.size(0) * val.size(1), -1)
    
    res = tensor.index_add(0, new_idx.view(-1), val).reshape([dim0_size, dim1_size, -1])

    if tensor.shape[-1] == 1:  # If selected element is scalar, restore original dim
        res = res.squeeze(-1)

    return res


def sample_subset(w, k, t=0.1):
    '''
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
    '''
    wg = gumbel_perturb(w)
    return continuous_topk(wg, k, t)


def logsumexp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
        
    return max_score + stable_vec.logsumexp(dim, keepdim=keepdim) # (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def clip_to_01(x: torch.Tensor):
    eps = torch.finfo(x.dtype).eps
    tiny = torch.finfo(x.dtype).tiny
    
    x_val = x.detach()
    
    cond_greater = (x_val > (1.0 - eps))
    diff_greater = (x_val - 1.0 + eps)
    
    cond_less = (x < tiny)
    diff_less = (tiny - x_val)
    
    x -= diff_greater * cond_greater
    x += diff_less * cond_less
    
    return x
    

def log1mexp(x):
    x -= torch.finfo(x.dtype).eps
    return torch.where(x > -0.693, (-torch.expm1(x)).log(), torch.log1p(-(x.exp()))) 


def gumbel_perturb(w):
    uniform_01 = torch.distributions.uniform.Uniform(1e-6, 1.0)
    # sample some gumbels
    u = uniform_01.sample(w.size()).cuda()
    g = -torch.log(-torch.log(u))
    w = w + g
    return w


def continuous_topk(w, k, t):
    khot_list = []
    onehot_approx = torch.zeros_like(w)
    
    for i in range(k):
        khot_mask = torch.clamp(1.0 - onehot_approx, min=1e-6)
        w += torch.log(khot_mask)
        onehot_approx = F.softmax(w / t, dim=-1)
        khot_list.append(onehot_approx)
        
    return torch.stack(khot_list, dim=0)


def stripe(x, n, w, offset=(0, 0), horizontal=1):
    r"""
    Returns a diagonal stripe of the tensor.
    Args:
        x (~torch.Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 1 if returns a horizontal stripe; 0 otherwise.
    Returns:
        a diagonal stripe of the tensor.
    Examples:
        >>> x = torch.arange(25).view(5, 5)
        >>> x
        tensor([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]])
        >>> stripe(x, 2, 3)
        tensor([[0, 1, 2],
                [6, 7, 8]])
        >>> stripe(x, 2, 3, (1, 1))
        tensor([[ 6,  7,  8],
                [12, 13, 14]])
        >>> stripe(x, 2, 3, (1, 1), 0)
        tensor([[ 6, 11, 16],
                [12, 17, 22]])
    """
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if horizontal == 1 else seq_len) * numel
    
    return x.as_strided(
        size=(n, w, *x.shape[2:]), 
        stride=stride,
        storage_offset=(offset[0]*seq_len+offset[1])*numel
    )


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
