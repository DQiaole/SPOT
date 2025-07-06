import torch
from inference.kv_memory_store import KeyValueMemoryStore
import math
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    FLASH_AVAIABLE = True
    print("flash attention is available!")
except:
    FLASH_AVAIABLE = False
    print("flash attention is not available!")


def get_similarity(mk, ms, qk, qe):
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x [N]    - Memory keys
    # ms: B x  1 x [N]    - Memory shrinkage
    # qk: B x CK x [HW/P] - Query keys
    # qe: B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened
    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None

    if qe is not None:
        # See appendix for derivation
        # or you can just trust me ヽ(ー_ー )ノ
        mk = mk.transpose(1, 2)
        a_sq = (mk.pow(2) @ qe)
        two_ab = 2 * (mk @ (qk * qe))
        b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
        similarity = (-a_sq+two_ab-b_sq)
    else:
        # similar to STCN if we don't have the selection term
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        two_ab = 2 * (mk.transpose(1, 2) @ qk)
        similarity = (-a_sq+two_ab)

    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)   # B*N*HW
    else:
        similarity = similarity / math.sqrt(CK)   # B*N*HW

    return similarity


def do_softmax(similarity, top_k=None, inplace=False, return_usage=False):
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=1)

        x_exp = values.exp_()
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp) # B*N*HW
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp) # B*N*HW
    else:
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(similarity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity


class MemoryManager:
    """
    Manages all three memory stores and the transition between working/long-term memory
    """

    def __init__(self, config):
        self.cfg = config
        # top_k for softmax
        self.top_k = getattr(config, 'top_k', None)

        self.max_mt_frames = config.max_mid_term_frames
        self.min_mt_frames = config.min_mid_term_frames

        # dimensions will be inferred from input later
        self.CK = self.CV = None
        self.H = self.W = None

        self.work_mem = KeyValueMemoryStore(count_usage=False)

        self.reset_config = True

    def _readout(self, affinity, v):
        # this function is for a single object group
        return v @ affinity

    def match_memory(self, query_key):
        # query_key: B x C^k x H x W
        h, w = query_key.shape[-2:]
        scale = query_key.shape[1] ** -0.5

        query_key = query_key.flatten(start_dim=2)

        """
        Memory readout using keys
        """

        if self.work_mem.engaged():
            # No long-term memory
            memory_key = self.work_mem.key
            if FLASH_AVAIABLE:
                query_key = query_key.permute(0, 2, 1).unsqueeze(2).bfloat16()
                memory_key = memory_key.permute(0, 2, 1).unsqueeze(2).bfloat16()
                memory_value = self.work_mem.value.permute(0, 2, 1).unsqueeze(2).bfloat16()
                memory1 = flash_attn_func(query_key, memory_key, memory_value[:, :, :, :memory_key.shape[-1]])
                memory2 = flash_attn_func(query_key, memory_key, memory_value[:, :, :, memory_key.shape[-1]:])
                all_readout_mem = torch.cat([memory1, memory2], dim=-1).squeeze(2).permute(0, 2, 1)
            else:
                similarity = torch.einsum('b c l, b c t -> b t l', query_key, memory_key) * scale

                # if self.enable_long_term:
                affinity, usage = do_softmax(similarity, inplace=True, top_k=self.top_k, return_usage=True)

                all_memory_value = self.work_mem.value
                work_usage = usage
                # Record memory usage for working memory
                self.work_mem.update_usage(work_usage.flatten())
                all_readout_mem = self._readout(affinity, all_memory_value)
            # else:
            #     raise NotImplementedError
        else:
            # No working-term memory
            return None

        return all_readout_mem.view(all_readout_mem.shape[0], -1, h, w)

    def add_memory(self, key, value):
        # key: 1*C*H*W
        # value: 1*C*H*W
        if self.H is None or self.reset_config:
            self.reset_config = False
            self.H, self.W = key.shape[-2:]
            self.HW = self.H * self.W
            # if self.enable_long_term:
            # convert from num. frames to num. nodes
            self.min_work_elements = self.min_mt_frames * self.HW
            self.max_work_elements = self.max_mt_frames * self.HW

        # key:   1*C*N
        # value: 1*C*N
        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)

        self.CK = key.shape[1]
        self.CV = value.shape[1]

        self.work_mem.add(key, value)

        # long-term memory cleanup
        # if self.enable_long_term:
        # Do memory compressed if needed
        if self.work_mem.size >= self.max_work_elements:
            self.compress_features()

    def compress_features(self):
        # remove consolidated working memory
        self.work_mem.sieve_by_range(0, -self.min_work_elements)
        

