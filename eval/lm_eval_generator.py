"""lm-evaluation-harness adapter for the Fast-dLLM batch generator."""

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from lm_eval.api.model import LM

from eval.generate import FastdLLMGenerator


CACHE_VERSION = "fixed-batch-repetition-v2"


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _truncate_at_stops(text: str, stops: list[str]) -> str:
    stop_positions = [text.find(stop) for stop in stops if stop and stop in text]
    return text[: min(stop_positions)] if stop_positions else text


class DLLMLM(LM):
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        batch_size: int = 8,
        threshold: float = 0.9,
        default_max_new_tokens: int = 2048,
        use_prefix_cache: bool = True,
        use_block_cache: bool = True,
        bd_size: int = 32,
        small_block_size: int = 8,
        seed: int = 0,
        repetition_penalty: float = 1.0,
        repetition_penalty_tasks: list[str] | None = None,
        generator: FastdLLMGenerator | None = None,
        generation_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model_path = os.path.abspath(model_path) if os.path.isdir(model_path) else model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._batch_size = int(batch_size)
        self.threshold = float(threshold)
        self.default_max_new_tokens = int(default_max_new_tokens)
        self.use_prefix_cache = bool(use_prefix_cache)
        self.use_block_cache = bool(use_block_cache)
        self.bd_size = int(bd_size)
        self.small_block_size = int(small_block_size)
        self.seed = int(seed)
        self.repetition_penalty = float(repetition_penalty)
        self.repetition_penalty_tasks = (
            tuple(repetition_penalty_tasks)
            if repetition_penalty_tasks is not None else None
        )
        self.generation_overrides = dict(generation_overrides or {})
        allowed_overrides = {"max_new_tokens", "temperature", "top_p"}
        if self.generation_overrides.keys() - allowed_overrides:
            raise ValueError("Unsupported generation override")
        self.generator = generator or FastdLLMGenerator(model_path=self.model_path)
        self.tokenizer = self.generator.tokenizer

        self.config = {
            "cache_version": CACHE_VERSION,
            "model_path": self.model_path,
            "batch_size": self._batch_size,
            "threshold": self.threshold,
            "default_max_new_tokens": self.default_max_new_tokens,
            "use_prefix_cache": self.use_prefix_cache,
            "use_block_cache": self.use_block_cache,
            "bd_size": self.bd_size,
            "small_block_size": self.small_block_size,
            "seed": self.seed,
            "repetition_penalty": self.repetition_penalty,
            "repetition_penalty_tasks": self.repetition_penalty_tasks,
        }
        self.config_hash = _json_hash(self.config)
        if self.generation_overrides:
            self.config["generation_overrides"] = self.generation_overrides
            self.config_hash = _json_hash(self.config)
        self.cache_path = self.output_dir / "generation_cache.jsonl"
        self.debug_path = self.output_dir / "input_debug.jsonl"
        self._cache = self._load_cache()
        self._debug_counts = self._load_debug_counts()

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        return self.generator.device

    @property
    def max_gen_toks(self) -> int:
        return self.default_max_new_tokens

    @property
    def max_length(self) -> int:
        return 4096

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def tokenizer_name(self) -> str:
        return self.model_path

    def chat_template(self, chat_template: bool | str = False) -> str | None:
        return self.tokenizer.chat_template

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def get_model_info(self) -> dict[str, Any]:
        return {"model_path": self.model_path, "generation_config": self.config}

    def loglikelihood(self, requests):
        raise NotImplementedError("The selected full-eval tasks use generate_until")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("The selected full-eval tasks use generate_until")

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        cache = {}
        if not self.cache_path.exists():
            return cache
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("config_hash") == self.config_hash and "key" in item:
                cache[item["key"]] = item
        return cache

    def _load_debug_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        if not self.debug_path.exists():
            return counts
        for line in self.debug_path.read_text(encoding="utf-8").splitlines():
            try:
                counts[json.loads(line)["task"]] += 1
            except (json.JSONDecodeError, KeyError):
                continue
        return counts

    def _request_key(self, task: str, context: str, gen_kwargs: dict) -> str:
        return _json_hash(
            {
                "config_hash": self.config_hash,
                "task": task,
                "context": context,
                "gen_kwargs": gen_kwargs,
            }
        )

    def _append_cache(self, item: dict[str, Any]) -> None:
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            handle.flush()
        self._cache[item["key"]] = item

    def _write_debug(self, item: dict[str, Any]) -> None:
        task = item["task"]
        if self._debug_counts[task] >= 2:
            return
        token_ids = self.tokenizer([item["context"]]).input_ids[0]
        record = {
            "task": task,
            "config_hash": self.config_hash,
            "context": item["context"],
            "token_ids": token_ids,
            "decoded_input": self.tokenizer.decode(token_ids),
            "raw_generation": item["raw_generation"],
            "completion": item["completion"],
            "gen_kwargs": item["gen_kwargs"],
        }
        with self.debug_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._debug_counts[task] += 1

    def generate_until(self, requests) -> list[str]:
        outputs: list[str | None] = [None] * len(requests)
        pending: list[dict[str, Any]] = []

        def task_repetition_penalty(task: str) -> float:
            scopes = self.repetition_penalty_tasks
            if scopes is None:
                return self.repetition_penalty
            if any(
                task == scope or task.startswith(scope + "_") for scope in scopes
            ):
                return self.repetition_penalty
            return 1.0

        for index, request in enumerate(requests):
            context, gen_kwargs = request.args
            gen_kwargs = dict(gen_kwargs or {})
            task = request.task_name or "unknown"
            key = self._request_key(task, context, gen_kwargs)
            cached = self._cache.get(key)
            if cached is not None:
                outputs[index] = cached["completion"]
                self._write_debug(cached)
                continue
            pending.append(
                {
                    "index": index,
                    "task": task,
                    "context": context,
                    "gen_kwargs": gen_kwargs,
                    "key": key,
                }
            )

        groups: dict[
            tuple[int, float, float, float], list[dict[str, Any]]
        ] = defaultdict(list)
        for item in pending:
            kwargs = item["gen_kwargs"]
            if kwargs.get("do_sample", False) and "temperature" not in self.generation_overrides:
                raise ValueError("Sampling requires an explicit temperature override")
            max_new_tokens = int(
                kwargs.get(
                    "max_gen_toks",
                    kwargs.get("max_new_tokens", self.default_max_new_tokens),
                )
            )
            temperature = float(kwargs.get("temperature", 0.0))
            top_p = float(kwargs.get("top_p", 0.95))
            if temperature != 0.0 and "temperature" not in self.generation_overrides:
                raise ValueError("Full evaluation requires temperature=0 unless explicitly overridden")
            max_new_tokens = int(self.generation_overrides.get("max_new_tokens", max_new_tokens))
            temperature = float(self.generation_overrides.get("temperature", temperature))
            top_p = float(self.generation_overrides.get("top_p", top_p))
            repetition_penalty = task_repetition_penalty(item["task"])
            groups[(max_new_tokens, temperature, top_p, repetition_penalty)].append(item)

        for (max_new_tokens, temperature, top_p, repetition_penalty), group in groups.items():
            group.sort(
                key=lambda item: len(self.tokenizer([item["context"]]).input_ids[0])
            )
            for start in range(0, len(group), self._batch_size):
                chunk = group[start : start + self._batch_size]
                raw_outputs = self.generator.batch_generate(
                    [item["context"] for item in chunk],
                    batch_size=len(chunk),
                    max_new_tokens=max_new_tokens,
                    threshold=self.threshold,
                    use_prefix_cache=self.use_prefix_cache,
                    use_block_cache=self.use_block_cache,
                    bd_size=self.bd_size,
                    repetition_penalty=repetition_penalty,
                    small_block_size=self.small_block_size,
                    temperature=temperature,
                    top_p=top_p,
                )
                for item, raw in zip(chunk, raw_outputs):
                    stops = item["gen_kwargs"].get("until", [])
                    if isinstance(stops, str):
                        stops = [stops]
                    completion = _truncate_at_stops(raw, list(stops))
                    cache_item = {
                        **item,
                        "config_hash": self.config_hash,
                        "raw_generation": raw,
                        "completion": completion,
                        "max_new_tokens": max_new_tokens,
                    }
                    self._append_cache(cache_item)
                    self._write_debug(cache_item)
                    outputs[item["index"]] = completion
                    self.cache_hook.add_partial(
                        "generate_until",
                        (item["context"], item["gen_kwargs"]),
                        completion,
                    )

        if any(output is None for output in outputs):
            raise RuntimeError("Missing lm-eval generations")
        return [str(output) for output in outputs]
