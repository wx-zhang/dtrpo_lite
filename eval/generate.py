"""Fast-dLLM generation for singleton-reference and batched evaluation.

``FastdLLMGenerator.generate`` intentionally accepts exactly one prompt and
preserves the original singleton generation path. Batched callers must use
``FastdLLMGenerator.batch_generate``; it sorts all requests by token length and
keeps prefix/block-cache decisions independent for every sample.
"""

import types
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoTokenizer, DynamicCache

from model.modeling_fastdllm import Fast_dLLM_QwenForCausalLM

FAST_DLLM_MASK_ID = 151665
FAST_DLLM_STOP_TOKEN = 151645


class FastdLLMGenerator:
    def __init__(
        self,
        model_path: str = "Efficient-Large-Model/Fast_dLLM_v2_7B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = Fast_dLLM_QwenForCausalLM.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=True
        )
        self.model.eval()
        self.model.mdm_sample = types.MethodType(_batch_sample, self.model)

        self.device = device
        self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, fix_mistral_regex=True
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 2048,
        mask_id: int = FAST_DLLM_MASK_ID,
        threshold: float = 0.9,
        use_block_cache: bool = True,
        small_block_size: int = 8,
        bd_size: int = 32,
        temperature: float = 0.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        use_prefix_cache: bool = True,
    ) -> List[str]:
        """Generate one singleton-reference completion.

        This is deliberately not a batch API. Use :meth:`batch_generate` for
        two or more prompts.
        """
        assert len(prompts) == 1, (
            "FastdLLMGenerator.generate() is the singleton reference API and "
            "requires exactly one prompt; use batch_generate() for batched inference"
        )
        if use_block_cache and not use_prefix_cache:
            raise ValueError("block cache requires prefix cache")
        if not use_prefix_cache:
            return self.batch_generate(
                prompts,
                batch_size=1,
                max_new_tokens=max_new_tokens,
                mask_id=mask_id,
                threshold=threshold,
                use_prefix_cache=False,
                use_block_cache=False,
                small_block_size=small_block_size,
                bd_size=bd_size,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        output: List[str] = [None] * len(prompts)  # type: ignore[list-item]

        # Sort by length so the batch is roughly uniform; remember original order.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        prompts_sorted = [prompts[i] for i in order]

        batched_input_ids = []
        seq_len: List[int] = []
        max_len = 0
        min_len = 1 << 30
        for p in prompts_sorted:
            ids = self.tokenizer([p], return_tensors="pt").input_ids.to(self.device)
            batched_input_ids.append(ids)
            max_len = max(max_len, ids.shape[1])
            min_len = min(min_len, ids.shape[1])
            seq_len.append(ids.shape[1])

        # Right-pad with the mask token so all rows align to max_len.
        batched_input_ids = [
            torch.cat(
                [
                    ids,
                    torch.full(
                        (1, max_len - ids.shape[1]),
                        mask_id,
                        dtype=torch.long,
                        device=self.device,
                    ),
                ],
                dim=1,
            )
            for ids in batched_input_ids
        ]
        batched_input_ids = torch.cat(batched_input_ids, dim=0)

        generated_ids = self.model.mdm_sample(
            batched_input_ids,
            tokenizer=self.tokenizer,
            block_size=bd_size,
            small_block_size=small_block_size,
            max_new_tokens=max_new_tokens,
            mask_id=mask_id,
            min_len=min_len,
            seq_len=torch.tensor(seq_len, device=self.device),
            use_block_cache=use_block_cache,
            threshold=threshold,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        for batch_pos, orig_idx in enumerate(order):
            decoded = self.tokenizer.decode(
                generated_ids[batch_pos][seq_len[batch_pos] : seq_len[batch_pos] + max_new_tokens],
                skip_special_tokens=True,
            )
            output[orig_idx] = decoded

        return output

    @torch.no_grad()
    def batch_generate(
        self,
        prompts: List[str],
        *,
        batch_size: int,
        max_new_tokens: int = 2048,
        mask_id: int = FAST_DLLM_MASK_ID,
        threshold: float = 0.9,
        use_prefix_cache: bool = True,
        use_block_cache: bool = True,
        small_block_size: int = 8,
        bd_size: int = 32,
        temperature: float = 0.0,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> List[str]:
        """Generate globally token-sorted batches and restore request order.

        This is the only public batch API. Each sample keeps an independent
        prefix frontier and full-block/partial-block cache decision.
        """
        if not prompts:
            return []
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if use_block_cache and not use_prefix_cache:
            raise ValueError("block cache requires prefix cache")
        if bd_size <= 0 or small_block_size <= 0 or bd_size % small_block_size:
            raise ValueError("bd_size must be divisible by small_block_size")
        _validate_repetition_penalty(repetition_penalty)

        requests = []
        for original_index, prompt in enumerate(prompts):
            input_ids = self.tokenizer(
                [prompt], return_tensors="pt"
            ).input_ids.squeeze(0).to(self.device)
            requests.append(
                _GenerationRequest(
                    original_index=original_index,
                    input_ids=input_ids,
                    generator=_make_generator(self.device, original_index),
                )
            )

        requests.sort(
            key=lambda request: (request.input_ids.numel(), request.original_index)
        )
        generated: dict[int, torch.Tensor] = {}
        for start in range(0, len(requests), batch_size):
            chunk = requests[start : start + batch_size]
            generated.update(
                _independent_batch_sample(
                    self.model,
                    chunk,
                    block_size=bd_size,
                    small_block_size=small_block_size,
                    max_new_tokens=max_new_tokens,
                    mask_id=mask_id,
                    stop_token=FAST_DLLM_STOP_TOKEN,
                    threshold=threshold,
                    use_prefix_cache=use_prefix_cache,
                    use_block_cache=use_block_cache,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
            )

        return [
            self.tokenizer.decode(generated[index], skip_special_tokens=True)
            for index in range(len(prompts))
        ]


# ─────────────────────────────────────────────────────────────────────────────
# The Fast-dLLM batch sampler — bound onto the model instance via MethodType.
# Logic copied verbatim from the dtrpo reference; only formatting tweaked.
# ─────────────────────────────────────────────────────────────────────────────


def _sample_with_top_p(logits, top_p=0.95, temperature=0.0, generator=None):
    if temperature == 0.0:
        p_1t = torch.nn.functional.softmax(logits, dim=-1)
        x_1 = torch.argmax(p_1t, dim=-1)
        return x_1, p_1t

    logits = logits / temperature
    p_1t = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(p_1t, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    p_1t = p_1t.masked_fill(indices_to_remove, 0.0)
    p_1t = p_1t / p_1t.sum(dim=-1, keepdim=True)
    x_1 = torch.multinomial(
        p_1t.view(-1, p_1t.size(-1)), num_samples=1, generator=generator
    ).view(
        p_1t.shape[:-1]
    )
    return x_1, p_1t


def _validate_repetition_penalty(repetition_penalty: float) -> None:
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be greater than 0")


def _apply_repetition_penalty(
    logits: torch.Tensor,
    token_histories: list[torch.Tensor],
    repetition_penalty: float,
) -> torch.Tensor:
    """Apply a sign-aware penalty to tokens already generated by each row.

    Positive logits are divided by the penalty and negative logits are
    multiplied by it. A token ID is penalized once regardless of how many
    times it already occurs in the completion.
    """
    _validate_repetition_penalty(repetition_penalty)
    if repetition_penalty == 1.0:
        return logits
    if logits.shape[0] != len(token_histories):
        raise ValueError("token_histories must contain one tensor per logits row")

    adjusted = logits.clone()
    vocab_size = adjusted.shape[-1]
    for row_idx, history in enumerate(token_histories):
        if history.numel() == 0:
            continue
        token_ids = torch.unique(history.to(device=logits.device, dtype=torch.long))
        token_ids = token_ids[(token_ids >= 0) & (token_ids < vocab_size)]
        if token_ids.numel() == 0:
            continue
        scores = adjusted[row_idx].index_select(-1, token_ids)
        scores = torch.where(
            scores < 0,
            scores * repetition_penalty,
            scores / repetition_penalty,
        )
        adjusted[row_idx].index_copy_(-1, token_ids, scores)
    return adjusted


def _completion_token_histories(
    sequences: torch.Tensor,
    prompt_lengths: list[int] | torch.Tensor,
    mask_id: int,
) -> list[torch.Tensor]:
    histories = []
    for row_idx, prompt_length in enumerate(prompt_lengths):
        start = int(prompt_length)
        history = sequences[row_idx, start:]
        histories.append(history[history.ne(mask_id)])
    return histories


@torch.no_grad()
def _batch_sample(
    self,
    input_ids,
    tokenizer,
    block_size,
    max_new_tokens,
    small_block_size,
    min_len,
    seq_len,
    mask_id=FAST_DLLM_MASK_ID,
    threshold=0.95,
    stop_token=FAST_DLLM_STOP_TOKEN,
    use_block_cache=False,
    top_p=0.95,
    temperature=0.0,
    repetition_penalty=1.0,
):
    _validate_repetition_penalty(repetition_penalty)

    num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
    batch_size = input_ids.shape[0]

    if min_len > block_size:
        output = self.forward(
            input_ids=input_ids[:, : (min_len // block_size * block_size)],
            use_cache=True,
            update_past_key_values=True,
            block_size=block_size,
        )
        logits, past_key_values = output.logits, output.past_key_values
        if min_len % block_size == 0:
            predict_sample_idx = seq_len == min_len
            predict_logits = logits[predict_sample_idx, -1:, :]
            next_token = predict_logits.argmax(dim=-1)
            if input_ids.shape[1] <= min_len:
                input_ids = torch.cat([input_ids, next_token], dim=1)
            else:
                input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
    else:
        past_key_values = None

    seq_block_idx = seq_len // block_size
    finished_flag = torch.zeros((batch_size,), device=self.device, dtype=torch.bool)

    start_block_idx = min_len // block_size
    num_small_blocks = block_size // small_block_size

    sample_indices = torch.arange(batch_size, device=self.device)
    finished_samples = {}
    for block_idx in range(start_block_idx, num_blocks):
        if finished_flag.all():
            break
        if (seq_block_idx == block_idx).all():
            x_init = mask_id * torch.ones(
                (input_ids.shape[0], block_size - input_ids.shape[1] % block_size),
                device=self.device,
                dtype=torch.long,
            )
            x_init = torch.cat([input_ids, x_init], dim=1)
            input_ids = x_init
        else:
            x_init = input_ids[:, : (block_idx + 1) * block_size]

        x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
        x_t = x_init.clone()
        block_past_key_values = None
        while True:
            mask_idx = x_t[:, -block_size:] == mask_id
            if mask_idx.sum() == 0:
                for sample_idx in range(x_t.shape[0]):
                    if (
                        finished_flag[sample_idx]
                        and seq_len[sample_idx] < (block_idx + 1) * block_size
                    ):
                        stop_token_idx = (
                            x_t[sample_idx, seq_len[sample_idx]:] == stop_token
                        ).nonzero()[0][0]
                        x_t[sample_idx, seq_len[sample_idx] + stop_token_idx + 1:] = (
                            tokenizer.pad_token_id
                        )
                if finished_flag.all():
                    break
                output = self.forward(
                    input_ids=x_t[:, -block_size:],
                    use_cache=True,
                    past_key_values=past_key_values,
                    update_past_key_values=True,
                    block_size=block_size,
                )
                logits, past_key_values = output.logits, output.past_key_values
                boundary_logits = _apply_repetition_penalty(
                    logits[:, -1:, :],
                    _completion_token_histories(x_t, seq_len, mask_id),
                    repetition_penalty,
                )
                next_token = boundary_logits.argmax(dim=-1)
                next_token[finished_flag] = tokenizer.pad_token_id
                x_t = torch.cat([x_t, next_token], dim=1)
                break

            for small_block_idx in range(num_small_blocks):
                small_block_start_idx = small_block_idx * small_block_size
                small_block_end_idx = small_block_start_idx + small_block_size
                start = -block_size + small_block_start_idx
                end = (
                    None
                    if block_size == small_block_end_idx
                    else -block_size + small_block_end_idx
                )
                while True:
                    mask_idx = x_t[:, -block_size:] == mask_id
                    if mask_idx[:, start:end].sum() == 0:
                        break

                    if use_block_cache:
                        if (
                            block_past_key_values is None
                            or (x_t[:, -block_size + small_block_start_idx] == mask_id).any()
                        ):
                            output = self.forward(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                use_block_cache=True,
                            )
                            logits = output.logits
                            block_past_key_values = output.block_past_key_values
                            logits = torch.cat(
                                [logits[:, :1, :], logits[:, :-1, :]], dim=1
                            )
                            logits = logits[:, start:end]
                        else:
                            logits = self.forward(
                                input_ids=x_t[:, start:end],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                use_block_cache=True,
                                block_past_key_values=block_past_key_values,
                                replace_position=small_block_start_idx,
                            ).logits
                            logits = torch.cat(
                                [logits[:, :1, :], logits[:, :-1, :]], dim=1
                            )
                    else:
                        logits = self.forward(
                            input_ids=x_t[:, -block_size:],
                            use_cache=True,
                            past_key_values=past_key_values,
                            update_past_key_values=False,
                        ).logits
                        logits = torch.cat(
                            [logits[:, :1, :], logits[:, :-1, :]], dim=1
                        )
                        logits = logits[:, start:end]

                    logits = _apply_repetition_penalty(
                        logits,
                        _completion_token_histories(x_t, seq_len, mask_id),
                        repetition_penalty,
                    )
                    x_1, p_1t = _sample_with_top_p(
                        logits, top_p=top_p, temperature=temperature
                    )
                    x1_p = torch.squeeze(
                        torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)),
                        -1,
                    )
                    x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                    unmask_idx = x1_p > threshold
                    max_prob_idx = x1_p.argmax(dim=-1)
                    unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                    unmask_idx = unmask_idx & mask_idx[:, start:end]
                    x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                    finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)
                    finished_flag = finished_flag | finished_row_flags

        if input_ids.shape[1] == x_t.shape[1]:
            input_ids = x_t
        else:
            input_ids[:, : (block_idx + 1) * block_size] = x_t[:, :-1]
            if (seq_block_idx == block_idx).all():
                input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
            else:
                if input_ids.shape[1] <= (block_idx + 1) * block_size:
                    input_ids = x_t
                else:
                    input_ids[
                        seq_block_idx == block_idx, (block_idx + 1) * block_size
                    ] = x_t[
                        seq_block_idx == block_idx, (block_idx + 1) * block_size
                    ]
        seq_block_idx[seq_block_idx == block_idx] = block_idx + 1
        if finished_flag.any():
            for sample_idx in range(x_t.shape[0]):
                if finished_flag[sample_idx]:
                    original_idx = sample_indices[sample_idx].item()
                    finished_samples[original_idx] = (
                        x_t[sample_idx : sample_idx + 1].clone().squeeze(dim=0)
                    )
            sample_indices = sample_indices[~finished_flag]
            input_ids = input_ids[~finished_flag]
            seq_block_idx = seq_block_idx[~finished_flag]
            seq_len = seq_len[~finished_flag]
            x_t = x_t[~finished_flag]
            keep_indices = torch.nonzero(~finished_flag, as_tuple=False).squeeze(-1)
            if past_key_values is not None:
                past_key_values.batch_select_indices(keep_indices)
            finished_flag = finished_flag[~finished_flag]

    if len(finished_samples) < batch_size:
        for sample_idx in range(x_t.shape[0]):
            original_idx = sample_indices[sample_idx].item()
            finished_samples[original_idx] = (
                x_t[sample_idx : sample_idx + 1].clone().squeeze(dim=0)
            )

    assert len(finished_samples) == batch_size
    return finished_samples


@dataclass
class _GenerationRequest:
    original_index: int
    input_ids: torch.Tensor
    generator: torch.Generator


@dataclass
class _GenerationGroup:
    requests: list[_GenerationRequest]
    tokens: torch.Tensor
    past_key_values: Optional[DynamicCache]
    pending_tokens: torch.Tensor

    @property
    def frontier(self) -> int:
        return self.tokens.shape[1]


def _make_generator(device: str, offset: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(torch.initial_seed() + offset)
    return generator


def _cache_select(
    cache: Optional[DynamicCache], indices: torch.Tensor
) -> Optional[DynamicCache]:
    if cache is None:
        return None
    return DynamicCache(
        ddp_cache_data=[
            (
                cache[layer_idx][0].index_select(0, indices).clone(),
                cache[layer_idx][1].index_select(0, indices).clone(),
            )
            for layer_idx in range(len(cache))
        ]
    )


def _cache_concat(caches: list[DynamicCache]) -> DynamicCache:
    if len(caches) == 1:
        return caches[0]
    return DynamicCache(
        ddp_cache_data=[
            (
                torch.cat([cache[layer_idx][0] for cache in caches], dim=0),
                torch.cat([cache[layer_idx][1] for cache in caches], dim=0),
            )
            for layer_idx in range(len(caches[0]))
        ]
    )


def _cache_scatter(
    cache: DynamicCache,
    updates: list[tuple[torch.Tensor, DynamicCache]],
) -> DynamicCache:
    data = []
    for layer_idx in range(len(cache)):
        keys = cache[layer_idx][0].clone()
        values = cache[layer_idx][1].clone()
        for indices, update in updates:
            keys.index_copy_(0, indices, update[layer_idx][0])
            values.index_copy_(0, indices, update[layer_idx][1])
        data.append((keys, values))
    return DynamicCache(ddp_cache_data=data)


def _merge_groups(
    groups: list[_GenerationGroup], use_prefix_cache: bool
) -> _GenerationGroup:
    if len(groups) == 1:
        return groups[0]
    frontier = groups[0].frontier
    if any(group.frontier != frontier for group in groups):
        raise ValueError("cannot merge generation groups at different frontiers")
    past_key_values = None
    if use_prefix_cache and frontier:
        caches = [group.past_key_values for group in groups]
        if any(cache is None for cache in caches):
            raise ValueError("missing prefix cache at a nonzero frontier")
        past_key_values = _cache_concat(caches)  # type: ignore[arg-type]
    return _GenerationGroup(
        requests=[request for group in groups for request in group.requests],
        tokens=torch.cat([group.tokens for group in groups], dim=0),
        past_key_values=past_key_values,
        pending_tokens=torch.cat([group.pending_tokens for group in groups], dim=0),
    )


def _group_completion_token_histories(
    group: _GenerationGroup,
    current_block: torch.Tensor,
    row_indices: torch.Tensor,
    mask_id: int,
) -> list[torch.Tensor]:
    histories = []
    for row_idx in row_indices.tolist():
        request = group.requests[row_idx]
        generated_start = request.input_ids.numel()
        committed = group.tokens[row_idx, generated_start:]
        block_start = max(generated_start - group.frontier, 0)
        revealed = current_block[row_idx, block_start:]
        revealed = revealed[revealed.ne(mask_id)]
        histories.append(torch.cat([committed, revealed]))
    return histories


def _sample_logits(
    logits: torch.Tensor,
    requests: list[_GenerationRequest],
    row_indices: torch.Tensor,
    top_p: float,
    temperature: float,
    token_histories: list[torch.Tensor],
    repetition_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = _apply_repetition_penalty(
        logits, token_histories, repetition_penalty
    )
    if temperature == 0.0:
        return _sample_with_top_p(logits, top_p=top_p, temperature=temperature)

    sampled_tokens = []
    sampled_probs = []
    for local_idx, row_idx in enumerate(row_indices.tolist()):
        tokens, probs = _sample_with_top_p(
            logits[local_idx : local_idx + 1],
            top_p=top_p,
            temperature=temperature,
            generator=requests[row_idx].generator,
        )
        sampled_tokens.append(tokens)
        sampled_probs.append(probs)
    return torch.cat(sampled_tokens, dim=0), torch.cat(sampled_probs, dim=0)


def _reveal_from_logits(
    current_block: torch.Tensor,
    row_indices: torch.Tensor,
    small_start: int,
    small_end: int,
    logits: torch.Tensor,
    group: _GenerationGroup,
    mask_id: int,
    threshold: float,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
) -> None:
    token_histories = (
        _group_completion_token_histories(
            group, current_block, row_indices, mask_id
        )
        if repetition_penalty != 1.0
        else [current_block.new_empty(0) for _ in range(logits.shape[0])]
    )
    sampled, probs = _sample_logits(
        logits,
        group.requests,
        row_indices,
        top_p,
        temperature,
        token_histories,
        repetition_penalty,
    )
    confidence = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)

    rows = current_block.index_select(0, row_indices).clone()
    block_slice = rows[:, small_start:small_end]
    masked = block_slice.eq(mask_id)
    confidence = torch.where(masked, confidence, -torch.inf)
    reveal = confidence > threshold
    max_indices = confidence.argmax(dim=-1)
    reveal[torch.arange(reveal.shape[0], device=reveal.device), max_indices] = True
    reveal &= masked
    block_slice[reveal] = sampled[reveal]
    rows[:, small_start:small_end] = block_slice
    current_block.index_copy_(0, row_indices, rows)


def _shift_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)


def _denoise_block(
    model,
    group: _GenerationGroup,
    current_block: torch.Tensor,
    *,
    block_size: int,
    small_block_size: int,
    mask_id: int,
    threshold: float,
    use_prefix_cache: bool,
    use_block_cache: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> torch.Tensor:
    block_cache: Optional[DynamicCache] = None

    for small_start in range(0, block_size, small_block_size):
        small_end = small_start + small_block_size
        while True:
            active_mask = current_block[:, small_start:small_end].eq(mask_id).any(dim=1)
            if not active_mask.any():
                break
            active_rows = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)

            if not use_prefix_cache:
                full_input = torch.cat(
                    [
                        group.tokens.index_select(0, active_rows),
                        current_block.index_select(0, active_rows),
                    ],
                    dim=1,
                )
                output = model(
                    input_ids=full_input,
                    use_cache=False,
                    block_size=block_size,
                )
                absolute_start = group.frontier + small_start
                logits = _shift_logits(output.logits)[
                    :, absolute_start : absolute_start + small_block_size
                ]
                _reveal_from_logits(
                    current_block,
                    active_rows,
                    small_start,
                    small_end,
                    logits,
                    group,
                    mask_id,
                    threshold,
                    top_p,
                    temperature,
                    repetition_penalty,
                )
                continue

            if not use_block_cache:
                output = model(
                    input_ids=current_block.index_select(0, active_rows),
                    use_cache=True,
                    past_key_values=_cache_select(group.past_key_values, active_rows),
                    update_past_key_values=False,
                    block_size=block_size,
                )
                logits = _shift_logits(output.logits)[:, small_start:small_end]
                _reveal_from_logits(
                    current_block,
                    active_rows,
                    small_start,
                    small_end,
                    logits,
                    group,
                    mask_id,
                    threshold,
                    top_p,
                    temperature,
                    repetition_penalty,
                )
                continue

            if block_cache is None:
                output = model(
                    input_ids=current_block,
                    use_cache=True,
                    past_key_values=group.past_key_values,
                    update_past_key_values=False,
                    block_size=block_size,
                    use_block_cache=True,
                )
                block_cache = output.block_past_key_values
                logits = _shift_logits(output.logits)[
                    :, small_start:small_end
                ].index_select(0, active_rows)
                _reveal_from_logits(
                    current_block,
                    active_rows,
                    small_start,
                    small_end,
                    logits,
                    group,
                    mask_id,
                    threshold,
                    top_p,
                    temperature,
                    repetition_penalty,
                )
                continue

            anchor_is_masked = current_block[:, small_start].eq(mask_id)
            full_rows = torch.nonzero(
                active_mask & anchor_is_masked, as_tuple=False
            ).squeeze(-1)
            partial_rows = torch.nonzero(
                active_mask & ~anchor_is_masked, as_tuple=False
            ).squeeze(-1)
            updates: list[tuple[torch.Tensor, DynamicCache]] = []

            if full_rows.numel():
                output = model(
                    input_ids=current_block.index_select(0, full_rows),
                    use_cache=True,
                    past_key_values=_cache_select(group.past_key_values, full_rows),
                    update_past_key_values=False,
                    block_size=block_size,
                    use_block_cache=True,
                )
                full_logits = _shift_logits(output.logits)[
                    :, small_start:small_end
                ]
                _reveal_from_logits(
                    current_block,
                    full_rows,
                    small_start,
                    small_end,
                    full_logits,
                    group,
                    mask_id,
                    threshold,
                    top_p,
                    temperature,
                    repetition_penalty,
                )
                updates.append((full_rows, output.block_past_key_values))

            if partial_rows.numel():
                output = model(
                    input_ids=current_block.index_select(
                        0, partial_rows
                    )[:, small_start:small_end],
                    use_cache=True,
                    past_key_values=_cache_select(
                        group.past_key_values, partial_rows
                    ),
                    update_past_key_values=False,
                    block_size=block_size,
                    use_block_cache=True,
                    block_past_key_values=_cache_select(block_cache, partial_rows),
                    replace_position=small_start,
                )
                partial_logits = _shift_logits(output.logits)
                _reveal_from_logits(
                    current_block,
                    partial_rows,
                    small_start,
                    small_end,
                    partial_logits,
                    group,
                    mask_id,
                    threshold,
                    top_p,
                    temperature,
                    repetition_penalty,
                )
                updates.append((partial_rows, output.block_past_key_values))

            block_cache = _cache_scatter(block_cache, updates)

    return current_block


def _build_current_block(
    group: _GenerationGroup, block_size: int, mask_id: int
) -> torch.Tensor:
    current = torch.full(
        (len(group.requests), block_size),
        mask_id,
        dtype=torch.long,
        device=group.tokens.device,
    )
    for row_idx, request in enumerate(group.requests):
        prompt_end = min(request.input_ids.numel(), group.frontier + block_size)
        if prompt_end > group.frontier:
            count = prompt_end - group.frontier
            current[row_idx, :count] = request.input_ids[group.frontier:prompt_end]
        pending = group.pending_tokens[row_idx]
        if pending.ge(0) and group.frontier >= request.input_ids.numel():
            current[row_idx, 0] = pending
    return current


def _finished_completion(
    sequence: torch.Tensor,
    request: _GenerationRequest,
    max_new_tokens: int,
    stop_token: int,
    block_size: int,
) -> Optional[torch.Tensor]:
    completion = sequence[request.input_ids.numel() :]
    prompt_remainder = request.input_ids.numel() % block_size
    generation_limit = (
        max_new_tokens
        if prompt_remainder == 0
        else max_new_tokens - prompt_remainder + 1
    )
    stop_positions = torch.nonzero(
        completion.eq(stop_token), as_tuple=False
    ).squeeze(-1)
    if stop_positions.numel():
        end = min(int(stop_positions[0].item()) + 1, generation_limit)
        return completion[:end].detach().cpu()
    if completion.numel() >= generation_limit:
        return completion[:generation_limit].detach().cpu()
    return None


def _commit_block(
    model,
    group: _GenerationGroup,
    current_block: torch.Tensor,
    outputs: dict[int, torch.Tensor],
    *,
    block_size: int,
    max_new_tokens: int,
    stop_token: int,
    use_prefix_cache: bool,
    repetition_penalty: float,
) -> Optional[_GenerationGroup]:
    completed = torch.cat([group.tokens, current_block], dim=1)
    continuing = []
    for row_idx, request in enumerate(group.requests):
        completion = _finished_completion(
            completed[row_idx], request, max_new_tokens, stop_token, block_size
        )
        if completion is None:
            continuing.append(row_idx)
        else:
            outputs[request.original_index] = completion

    if not continuing:
        return None

    rows = torch.tensor(continuing, dtype=torch.long, device=completed.device)
    continuing_requests = [group.requests[row_idx] for row_idx in continuing]
    continuing_tokens = completed.index_select(0, rows)
    continuing_block = current_block.index_select(0, rows)

    if use_prefix_cache:
        output = model(
            input_ids=continuing_block,
            use_cache=True,
            past_key_values=_cache_select(group.past_key_values, rows),
            update_past_key_values=True,
            block_size=block_size,
        )
        next_past = output.past_key_values
    else:
        output = model(
            input_ids=continuing_tokens,
            use_cache=False,
            block_size=block_size,
        )
        next_past = None

    token_histories = [
        continuing_tokens[row_idx, request.input_ids.numel() :]
        for row_idx, request in enumerate(continuing_requests)
    ]
    boundary_logits = _apply_repetition_penalty(
        output.logits[:, -1:, :],
        token_histories,
        repetition_penalty,
    )
    next_tokens = boundary_logits.argmax(dim=-1).squeeze(-1)

    keep = []
    for local_idx, request in enumerate(continuing_requests):
        sequence_with_boundary = torch.cat(
            [continuing_tokens[local_idx], next_tokens[local_idx : local_idx + 1]]
        )
        completion = _finished_completion(
            sequence_with_boundary, request, max_new_tokens, stop_token, block_size
        )
        if completion is None:
            keep.append(local_idx)
        else:
            outputs[request.original_index] = completion

    if not keep:
        return None

    keep_rows = torch.tensor(keep, dtype=torch.long, device=completed.device)
    return _GenerationGroup(
        requests=[continuing_requests[row_idx] for row_idx in keep],
        tokens=continuing_tokens.index_select(0, keep_rows),
        past_key_values=_cache_select(next_past, keep_rows),
        pending_tokens=next_tokens.index_select(0, keep_rows),
    )


def _initial_groups(
    model,
    requests: list[_GenerationRequest],
    block_size: int,
    use_prefix_cache: bool,
) -> dict[int, list[_GenerationGroup]]:
    buckets: dict[int, list[_GenerationRequest]] = defaultdict(list)
    for request in requests:
        frontier = request.input_ids.numel() // block_size * block_size
        buckets[frontier].append(request)

    groups: dict[int, list[_GenerationGroup]] = defaultdict(list)
    for frontier, bucket in buckets.items():
        if frontier:
            tokens = torch.stack(
                [request.input_ids[:frontier] for request in bucket], dim=0
            )
        else:
            tokens = torch.empty(
                (len(bucket), 0),
                dtype=torch.long,
                device=bucket[0].input_ids.device,
            )

        past_key_values = None
        pending = torch.full(
            (len(bucket),),
            -1,
            dtype=torch.long,
            device=tokens.device,
        )
        if use_prefix_cache and frontier:
            output = model(
                input_ids=tokens,
                use_cache=True,
                update_past_key_values=True,
                block_size=block_size,
            )
            past_key_values = output.past_key_values
            exact_boundary_rows = torch.tensor(
                [request.input_ids.numel() == frontier for request in bucket],
                dtype=torch.bool,
                device=tokens.device,
            )
            if exact_boundary_rows.any():
                pending[exact_boundary_rows] = output.logits[
                    exact_boundary_rows, -1, :
                ].argmax(dim=-1)

        groups[frontier].append(
            _GenerationGroup(
                requests=bucket,
                tokens=tokens,
                past_key_values=past_key_values,
                pending_tokens=pending,
            )
        )
    return groups


@torch.no_grad()
def _independent_batch_sample(
    model,
    requests: list[_GenerationRequest],
    *,
    block_size: int,
    small_block_size: int,
    max_new_tokens: int,
    mask_id: int,
    stop_token: int,
    threshold: float,
    use_prefix_cache: bool,
    use_block_cache: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.0,
) -> dict[int, torch.Tensor]:
    _validate_repetition_penalty(repetition_penalty)

    groups = _initial_groups(model, requests, block_size, use_prefix_cache)
    outputs: dict[int, torch.Tensor] = {}

    while groups:
        frontier = min(groups)
        group = _merge_groups(groups.pop(frontier), use_prefix_cache)
        current_block = _build_current_block(group, block_size, mask_id)
        current_block = _denoise_block(
            model,
            group,
            current_block,
            block_size=block_size,
            small_block_size=small_block_size,
            mask_id=mask_id,
            threshold=threshold,
            use_prefix_cache=use_prefix_cache,
            use_block_cache=use_block_cache,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        next_group = _commit_block(
            model,
            group,
            current_block,
            outputs,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
            stop_token=stop_token,
            use_prefix_cache=use_prefix_cache,
            repetition_penalty=repetition_penalty,
        )
        if next_group is not None:
            groups[next_group.frontier].append(next_group)

    if len(outputs) != len(requests):
        raise RuntimeError(
            f"generation produced {len(outputs)} outputs for {len(requests)} requests"
        )
    return outputs
