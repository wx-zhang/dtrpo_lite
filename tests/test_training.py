"""CPU regressions for trajectory replay and preference-training fixes."""
import copy
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from trainer.distributed_dtrpo_trainer import (
    DTRPOReplayRecord, FastDLLMDistributedDTRPOTrainer,
)
from trainer.dpo_base import DLDPOTrainer, DataCollatorAndTokenizerForPreference
from trainer.trl_compat import maybe_extract_prompt, maybe_apply_chat_template


class ToyDiffusion(torch.nn.Module):
    """Small differentiable model with Fast-dLLM's noisy-input/label contract."""
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(torch.randn(8, 8))
        self.model = SimpleNamespace(bd_size=32)

    def forward(self, input_ids, attention_mask, labels, compute_loss=False,
                precomputed_noisy_input_ids=None):
        assert self.training
        assert input_ids.dtype == labels.dtype == torch.long
        assert not compute_loss
        noisy = precomputed_noisy_input_ids
        if noisy is None:
            selected = (torch.rand(input_ids.shape) < 0.6) & (labels != -100)
            noisy = input_ids.masked_fill(selected, 7)
        labels[noisy != 7] = -100
        return SimpleNamespace(logits=self.table[noisy] + 0.1 * self.table[input_ids],
                               noisy_input_ids=noisy)


def make_trainer(samples=4, accumulation=1, microbatch=1):
    torch.manual_seed(17)
    trainer = object.__new__(FastDLLMDistributedDTRPOTrainer)
    trainer.is_encoder_decoder = False
    trainer.model = ToyDiffusion()
    trainer.ref_model = copy.deepcopy(trainer.model).requires_grad_(False).eval()
    with torch.no_grad():
        trainer.model.table.add_(torch.linspace(-0.2, 0.2, 64).reshape(8, 8))
    trainer.args = SimpleNamespace(mc_samples=samples, mc_batch_size=microbatch,
        logp_type="dtrpo", block_size=32, unmask_size=2, loss_type="sigmoid",
        label_smoothing=0.0, f_divergence_type="reverse_kl", rpo_alpha=None,
        disable_dropout=True)
    trainer.beta = 0.03
    trainer.label_smoothing = 0.0
    trainer.f_divergence_type = "reverse_kl"
    trainer.reference_free = trainer.use_weighting = trainer.aux_loss_enabled = False
    trainer.loss_type = ["sigmoid"]
    trainer.padding_value = 0
    trainer.max_length = 32
    trainer.truncation_mode = "keep_start"
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"),
        gather_for_metrics=lambda x: x, backward=lambda x: x.backward())
    trainer.optimizer = None
    trainer.current_gradient_accumulation_steps = accumulation
    trainer.compute_loss_context_manager = nullcontext
    trainer._prepare_context_parallel_inputs = lambda model, inputs: (nullcontext, inputs)
    trainer._prepare_inputs = lambda inputs: inputs
    trainer.store_metrics = lambda *args, **kwargs: None
    trainer._validate_distributed_dtrpo_config()
    return trainer


def preference_batch(rank=0):
    # Include padding and integer inputs narrower than int64.
    prompt = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32)
    chosen = torch.tensor([[3, 4, 5, 6], [4, 5, 0, 0]], dtype=torch.int32)
    rejected = torch.tensor([[5, 4, 3, 0], [3, 6, 5, 4]], dtype=torch.int32)
    if rank:
        chosen, rejected = rejected, chosen
    return dict(prompt_input_ids=prompt, prompt_attention_mask=(prompt != 0).long(),
        chosen_input_ids=chosen, chosen_attention_mask=(chosen != 0).long(),
        rejected_input_ids=rejected, rejected_attention_mask=(rejected != 0).long())


def direct_gradient(trainer, batch, seed, records=None):
    """Differentiate the full objective, replaying each trajectory separately."""
    prepared = trainer._prepare_preference_batch(batch)
    torch.manual_seed(seed)
    if records is None:
        records = trainer._collect_replay_records(trainer.model, prepared)
    scores = []
    nseq = prepared.input_ids.shape[0]
    for record in records:
        count = record.policy_logps.shape[1]
        noises = record.noisy_input_ids.reshape(nseq, count, -1)
        positions = record.unmask_positions.reshape(
            nseq, count, -1, record.unmask_positions.shape[-1])
        for i in range(count):
            scores.append(trainer._score_policy(trainer.model, prepared,
                noisy_input_ids=noises[:, i],
                unmask_positions=positions[:, i].reshape(record.unmask_positions.shape[0] // count, -1),
            )[0])
    policy = torch.stack(scores).mean(0)
    reference = torch.cat([r.reference_logps for r in records], dim=1).mean(1)
    ratios = policy - reference
    n = prepared.num_examples
    loss = -torch.nn.functional.logsigmoid(trainer.beta * (ratios[:n] - ratios[n:])).mean()
    scaled = loss / trainer.current_gradient_accumulation_steps
    scaled.backward()
    gradient = trainer.model.table.grad.clone()
    trainer.model.zero_grad(set_to_none=True)
    return scaled.detach(), gradient


def distributed_worker(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        trainer = make_trainer(samples=5, accumulation=2, microbatch=2)
        batch = preference_batch(rank)
        seed = 51 + rank
        expected_loss, expected = direct_gradient(trainer, batch, seed)
        dist.all_reduce(expected)
        expected /= world_size
        wrapped = DistributedDataParallel(trainer.model)
        torch.manual_seed(seed)
        actual_loss = trainer.training_step(wrapped, batch)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(trainer.model.table.grad, expected, rtol=1e-5, atol=1e-6)
        assert trainer.ref_model.table.grad is None
    finally:
        dist.destroy_process_group()


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_prepared_batch_masks_prompt_and_padding(self):
        trainer = make_trainer()
        prepared = trainer._prepare_preference_batch(preference_batch())
        self.assertEqual(prepared.num_examples, 2)
        self.assertEqual(prepared.input_ids.shape, (4, 32))
        self.assertEqual(prepared.input_ids.dtype, torch.long)
        self.assertEqual(prepared.labels.dtype, torch.long)
        self.assertEqual(prepared.input_ids[0, :8].tolist(), [1, 2, 3, 4, 5, 6, 0, 0])
        self.assertEqual(prepared.labels[0, :8].tolist(),
                         [-100, 3, 4, 5, 6, -100, -100, -100])
        self.assertTrue((prepared.attention_mask[0, 6:] == 0).all())
        self.assertTrue((prepared.labels[0, 6:] == -100).all())

    def test_standard_and_replay_receive_identical_prepared_inputs(self):
        for mode in ["keep_start", "keep_end"]:
            for length in [None, 4, 1]:
                for empty_prompt in [False, True]:
                    with self.subTest(mode=mode, length=length, empty_prompt=empty_prompt):
                        trainer = make_trainer(samples=1)
                        trainer.truncation_mode = mode
                        trainer.max_length = length
                        batch = preference_batch()
                        if empty_prompt:
                            batch["prompt_input_ids"] = batch["prompt_input_ids"][:, :0]
                            batch["prompt_attention_mask"] = batch["prompt_attention_mask"][:, :0]
                        calls = []
                        forward = trainer.model.forward
                        def record(input_ids, **kwargs):
                            calls.append(tuple(t.clone() for t in (
                                input_ids, kwargs["attention_mask"], kwargs["labels"])))
                            return forward(input_ids, **kwargs)
                        with patch.object(trainer.model, "forward", side_effect=record):
                            torch.manual_seed(73)
                            standard = trainer.concatenated_forward_with_random_masking(
                                trainer.model, batch, mc_samples=1, mc_batch_size=1,
                                mask_token_id=7)
                            prepared = trainer._prepare_preference_batch(batch)
                            labels = prepared.labels.clone()
                            torch.manual_seed(73)
                            replay, _, _ = trainer._score_policy(trainer.model, prepared)
                        self.assertEqual(len(calls), 2)
                        for standard_input, replay_input in zip(*calls):
                            torch.testing.assert_close(standard_input, replay_input)
                        torch.testing.assert_close(prepared.labels, labels)
                        torch.testing.assert_close(replay, torch.cat((
                            standard["chosen_logps"], standard["rejected_logps"])))

    def test_replay_gradient_matches_full_objective(self):
        cases = [(1, 1, 1), (4, 1, 1), (4, 3, 2), (5, 3, 2), (5, 3, 3), (5, 1, 8)]
        for samples, accumulation, microbatch in cases:
            for multiple_blocks in [False, True]:
                with self.subTest(samples=samples, accumulation=accumulation,
                                  microbatch=microbatch, multiple_blocks=multiple_blocks):
                    trainer = make_trainer(samples, accumulation, microbatch)
                    batch = preference_batch()
                    if multiple_blocks:
                        trainer.max_length = 64
                        for key in ["chosen_input_ids", "chosen_attention_mask",
                                    "rejected_input_ids", "rejected_attention_mask"]:
                            batch[key] = batch[key].repeat(1, 10)
                    loss, gradient = direct_gradient(trainer, batch, seed=91)
                    torch.manual_seed(91)
                    actual_loss = trainer.training_step(trainer.model, batch)
                    torch.testing.assert_close(actual_loss, loss)
                    torch.testing.assert_close(trainer.model.table.grad, gradient,
                                               rtol=1e-5, atol=1e-6)
                    self.assertIsNone(trainer.ref_model.table.grad)

    def test_mc_microbatch_controls_both_forward_passes(self):
        trainer = make_trainer(samples=5, microbatch=2)
        calls = []
        forward = trainer.model.forward
        def record(ids, **kwargs):
            calls.append((torch.is_grad_enabled(), ids.shape[0]))
            return forward(ids, **kwargs)
        with patch.object(trainer.model, "forward", side_effect=record):
            trainer.training_step(trainer.model, preference_batch())
        self.assertEqual(calls, [(False, 8), (False, 8), (False, 4),
                                 (True, 8), (True, 8), (True, 4)])

    def test_fixed_trajectories_are_invariant_to_replay_grouping(self):
        trainer = make_trainer(samples=5, accumulation=3)
        batch = preference_batch()
        trainer.max_length = 64
        for key in ["chosen_input_ids", "chosen_attention_mask",
                    "rejected_input_ids", "rejected_attention_mask"]:
            batch[key] = batch[key].repeat(1, 10)
        prepared = trainer._prepare_preference_batch(batch)
        records = trainer._collect_replay_records(trainer.model, prepared)
        expected_loss, expected_gradient = direct_gradient(trainer, batch, 17, records)
        nseq = prepared.input_ids.shape[0]
        for size in [1, 2, 3, 8]:
            grouped = []
            for start in range(0, len(records), size):
                chunk = records[start:start + size]
                grouped.append(DTRPOReplayRecord(
                    noisy_input_ids=torch.stack([r.noisy_input_ids for r in chunk], dim=1)
                        .reshape(-1, prepared.input_ids.shape[1]),
                    unmask_positions=torch.stack([
                        r.unmask_positions.reshape(nseq, -1, trainer.args.block_size)
                        for r in chunk], dim=1).reshape(-1, trainer.args.block_size),
                    policy_logps=torch.cat([r.policy_logps for r in chunk], dim=1),
                    reference_logps=torch.cat([r.reference_logps for r in chunk], dim=1),
                ))
            trainer.args.mc_batch_size = size
            trainer.model.zero_grad(set_to_none=True)
            with patch.object(trainer, "_collect_replay_records", return_value=grouped):
                actual_loss = trainer.training_step(trainer.model, batch)
            torch.testing.assert_close(actual_loss, expected_loss)
            torch.testing.assert_close(trainer.model.table.grad, expected_gradient,
                                       rtol=1e-5, atol=1e-6)

    def test_replay_rejects_nonpositive_mc_counts(self):
        for samples, size in [(0, 1), (1, 0), (1, -1)]:
            with self.subTest(samples=samples, microbatch=size):
                with self.assertRaises(ValueError):
                    make_trainer(samples=samples, microbatch=size)

    def test_two_rank_replay_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "gloo").as_uri()
            mp.spawn(distributed_worker, args=(2, rendezvous), nprocs=2, join=True)

    def test_mc_microbatch_average_including_remainder(self):
        trainer = make_trainer()
        trainer.is_encoder_decoder = False
        def forward(model, ref_model, ids, *args):
            values = ids.float().sum(1)
            return values, values + 1, {}, {}
        trainer._mc_forward_step = forward
        batch = preference_batch()
        results = []
        for size in [1, 2, 3, 5]:
            result = trainer.concatenated_forward_with_random_masking(
                trainer.model, batch, mc_samples=5, mc_batch_size=size, mask_token_id=7)
            results.append(result)
        for result in results[1:]:
            for key in ["chosen_logps", "rejected_logps", "chosen_logps_ref", "rejected_logps_ref"]:
                torch.testing.assert_close(result[key], results[0][key])

    def test_standard_vrpo_forward_without_selected_positions(self):
        trainer = make_trainer()
        trainer.args.logp_type = "vrpo"
        ids = torch.tensor([[1, 2, 3, 4], [1, 2, 5, 6]])
        mask = torch.tensor([[False, False, True, True]] * 2)
        policy, reference, info, _ = trainer._mc_forward_step(
            trainer.model, trainer.ref_model, ids, torch.ones_like(ids), ids, mask, 7)
        self.assertIsNone(info["unmask_positions"])
        self.assertTrue(torch.isfinite(policy).all())
        self.assertTrue(torch.isfinite(reference).all())

    def test_prompt_extraction_keeps_only_assistant_completion(self):
        prompt = [{"role": "user", "content": "Question"}]
        example = dict(prompt="Question",
            chosen=prompt + [{"role": "assistant", "content": "Correct"}],
            rejected=prompt + [{"role": "assistant", "content": "Wrong"}])
        extracted = maybe_extract_prompt(example)
        self.assertEqual(extracted["prompt"], prompt)
        self.assertEqual(extracted["chosen"], example["chosen"][-1:])
        class Tokenizer:
            def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
                value = "".join(f"<{m['role']}>{m['content']}|" for m in messages)
                return value + ("<assistant>" if add_generation_prompt else "")
        formatted = maybe_apply_chat_template(extracted, Tokenizer())
        self.assertEqual(formatted["chosen"], "Correct|")
        self.assertEqual(formatted["rejected"], "Wrong|")
        self.assertTrue(formatted["_is_chat_template_applied"])

    def test_collator_preserves_integer_empty_prompt_and_chat_flag(self):
        seen = []
        def tokenize(row, **kwargs):
            seen.append(kwargs)
            return dict(prompt_input_ids=[], chosen_input_ids=[1], rejected_input_ids=[2])
        collator = DataCollatorAndTokenizerForPreference({}, tokenize, pad_token_id=0)
        batch = collator([{"_is_chat_template_applied": True}])
        self.assertEqual(batch["prompt_input_ids"].dtype, torch.long)
        self.assertTrue(seen[0]["is_chat"])

    def test_supplied_tokenizer_is_preserved(self):
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=7)
        model = SimpleNamespace(config=SimpleNamespace(_name_or_path="unused"))
        args = SimpleNamespace(max_prompt_length=16, max_length=32)
        with patch("trainer.dpo_base.AutoTokenizer.from_pretrained") as load, \
             patch("trl.DPOTrainer.__init__", return_value=None) as parent:
            trainer = DLDPOTrainer(model, args=args, processing_class=tokenizer)
        load.assert_not_called()
        self.assertIs(parent.call_args.kwargs["processing_class"], tokenizer)
        self.assertEqual(trainer.padding_value, 0)


if __name__ == "__main__":
    unittest.main()
