import torch
from torch.optim import AdamW

import torch

def prune_weight_matrix_prunedlora(weight_A, weight_B, grad_A, grad_B, mask, k=5):
    """
    Structured pruning for LoRA matrices (A, B) with update step (PrunedLoRA).
    weight_A: torch.Tensor [r, d]  (LoRA A matrix)
    weight_B: torch.Tensor [d, r]  (LoRA B matrix)
    grad_A:   torch.Tensor [r, d]  (gradient wrt A)
    grad_B:   torch.Tensor [d, r]  (gradient wrt B)
    mask:     torch.BoolTensor [r] (which columns are active)
    k:        number of columns to prune
    """
    WB = weight_B.clone()
    WA = weight_A.clone()
    GB = grad_B.clone() if grad_B is not None else torch.zeros_like(WB)
    GA = grad_A.clone() if grad_A is not None else torch.zeros_like(WA)

    mask_new = mask.to(dtype=torch.bool).clone()
    WB[:, ~mask_new] = 0.0
    WA[~mask_new, :] = 0.0
    GB[:, ~mask_new] = 0.0
    GA[~mask_new, :] = 0.0

    candidate_indices = mask_new.nonzero(as_tuple=False).squeeze(1).tolist()
    if len(candidate_indices) <= k:
        return mask_new, WA, WB

    WB_sub = WB[:, candidate_indices]
    WA_sub = WA[candidate_indices, :]
    GB_sub = GB[:, candidate_indices]
    GA_sub = GA[candidate_indices, :]

    HB_sub = GB_sub.T @ GB_sub
    HA_sub = GA_sub @ GA_sub.T
    eps = 1e-6
    HB_inv = torch.linalg.pinv(HB_sub + eps * torch.eye(HB_sub.shape[0], device=HB_sub.device))
    HA_inv = torch.linalg.pinv(HA_sub + eps * torch.eye(HA_sub.shape[0], device=HA_sub.device))

    scores = []
    Wproj_B = GB_sub @ HB_inv
    Wproj_A = HA_inv @ GA_sub
    for local_i, j in enumerate(candidate_indices):
        rB_j = WB_sub[:, local_i] - Wproj_B[:, local_i]
        rA_j = WA_sub[local_i, :] - Wproj_A[local_i, :]
        HBjj_inv = HB_inv[local_i, local_i] + eps
        HAjj_inv = HA_inv[local_i, local_i] + eps
        score = 0.5 * (rB_j @ rB_j) / HBjj_inv + 0.5 * (rA_j @ rA_j) / HAjj_inv
        scores.append((score.item(), local_i, j))

    scores.sort(key=lambda x: x[0])
    prune_list = scores[:k]
    prune_global = [x[2] for x in prune_list]
    prune_local = [x[1] for x in prune_list]

    for local_i, j in zip(prune_local, prune_global):
        gB_j = GB_sub[:, local_i]
        gA_j = GA_sub[local_i, :]

        delta_B = - HB_inv @ gB_j
        delta_A = - gA_j @ HA_inv

        WB[:, mask_new] += delta_B.unsqueeze(1)
        WA[mask_new, :] += delta_A.unsqueeze(0)

        WB[:, j] = 0.0
        WA[j, :] = 0.0
        mask_new[j] = False

    return mask_new, WA, WB

def prune_weight_matrix_wanda(weight_A, weight_B, act_in, mask, k=5):
    mask_new = mask.clone()
    candidate_indices = mask_new.nonzero(as_tuple=False).squeeze(1).tolist()
    if len(candidate_indices) <= k:
        return mask_new, weight_A, weight_B
    act_proj = torch.norm(weight_A @ act_in.T, dim=1) / act_in.shape[0]
    weight_norm = torch.norm(weight_B, dim=0)
    scores = act_proj * weight_norm
    prune_idx = torch.argsort(scores)[:k]
    mask_new[prune_idx] = False
    weight_A[prune_idx, :] = 0.0
    weight_B[:, prune_idx] = 0.0
    return mask_new, weight_A, weight_B


def prune_weight_matrix_llmpruner(weight_A, weight_B, grad_A, grad_B, mask, k=5, variant="element1"):
    WA = weight_A.clone()
    WB = weight_B.clone()
    gA = (grad_A if grad_A is not None else torch.zeros_like(WA)).clone()
    gB = (grad_B if grad_B is not None else torch.zeros_like(WB)).clone()
    mask_new = mask.to(torch.bool).clone()
    if mask_new.sum().item() <= k:
        return mask_new, WA, WB
    WA[~mask_new, :] = 0.0
    WB[:, ~mask_new] = 0.0
    gA[~mask_new, :] = 0.0
    gB[:, ~mask_new] = 0.0
    if variant == "element1":
        elem_A = (gA * WA).abs()
        elem_B = (gB * WB).abs()
    elif variant == "element2":
        elem_A = (gA ** 2) * (WA ** 2)
        elem_B = (gB ** 2) * (WB ** 2)
    elif variant == "taylor2":
        gw_A = gA * WA
        gw_B = gB * WB
        elem_A = (gw_A - 0.5 * gw_A**2).abs()
        elem_B = (gw_B - 0.5 * gw_B**2).abs()
    else:
        raise ValueError(f"Unknown variant {variant}")
    score_A = elem_A.sum(dim=1)
    score_B = elem_B.sum(dim=0)
    scores = score_A + score_B
    candidate_indices = mask_new.nonzero(as_tuple=False).squeeze(1)
    cand_scores = scores[candidate_indices]
    _, order = torch.sort(cand_scores)
    prune_idx = candidate_indices[order[:k]]
    mask_new[prune_idx] = False
    WA[prune_idx, :] = 0.0
    WB[:, prune_idx] = 0.0
    return mask_new, WA, WB

class AdamWWithPruning(AdamW):
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, prune_every=100, prune_k=5, pruning_method="prunedlora", **kwargs):
        super().__init__(model.parameters(), lr=lr, betas=betas,
                         eps=eps, weight_decay=weight_decay, **kwargs)
        self.model = model
        self.prune_every = prune_every
        self.prune_k = prune_k
        self._step_count = 0
        self.pruning_method = pruning_method

    @torch.no_grad()
    def step(self, closure=None):
        for module in self.model.modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "mask"):
                for adapter_name in module.lora_A:
                    A = module.lora_A[adapter_name]
                    B = module.lora_B[adapter_name]
                    A.weight.data *= module.mask[:, None]
                    B.weight.data *= module.mask[None, :]
                    if A.weight.grad is not None:
                        A.weight.grad.data *= module.mask[:, None]
                    if B.weight.grad is not None:
                        B.weight.grad.data *= module.mask[None, :]
                    if self._step_count % self.prune_every == 0:
                        current_rank = int(module.mask.sum().item())
                        if current_rank > 64:
                            if self.pruning_method == "prunedlora":
                                new_mask, new_A, new_B = prune_weight_matrix_prunedlora(
                                    A.weight, B.weight, A.weight.grad, B.weight.grad, module.mask, k=self.prune_k)
                            elif self.pruning_method == "wanda":
                                new_mask, new_A, new_B = prune_weight_matrix_wanda(
                                    A.weight, B.weight, module.act_in, module.mask, k=self.prune_k)
                            elif self.pruning_method == "sparsegpt":
                                new_mask, new_A, new_B = prune_weight_matrix_sparsegpt(
                                    A.weight, B.weight, module.act_in, module.mask, k=self.prune_k)
                            elif self.pruning_method == "llmpruner":
                                new_mask, new_A, new_B = prune_weight_matrix_llmpruner(
                                    A.weight, B.weight, A.weight.grad, B.weight.grad, module.mask, k=self.prune_k)
                            else:
                                raise ValueError(f"Unknown pruning method: {self.pruning_method}")
                            module.mask = new_mask
                            A.weight.copy_(new_A)
                            B.weight.copy_(new_B)
                        else:
                            print(f"skip pruning {adapter_name} (rank={current_rank})")
        self._step_count += 1
        return super().step(closure)
