"""Lazy real GPU backend. Importing this module loads no external model/runtime."""
from __future__ import annotations

import importlib.metadata
import io
import platform
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

from .pure import canonical, reduce_readout, sha, encode_nonfinite_scalars


class QwenRuntime:
    """One-sequence exact frozen distribution; no retry or fallback backend."""

    def __init__(self, config: dict):
        observed = {"python": platform.python_version()}
        for name in ("torch", "transformers", "datasets", "accelerate",
                     "huggingface_hub", "safetensors"):
            observed[name] = importlib.metadata.version(name)
        expected = config["runtime"]["expected_versions"]
        if any(observed[name] != version for name, version in expected.items()):
            raise RuntimeError("Pinned generation runtime versions differ; no model loaded")
        import torch
        from huggingface_hub import HfApi, snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with native BF16 is required; no model loaded")
        self.device = torch.device(config["model"]["device"])
        self.config = config
        offline = config.get("local_files_only", False)
        if not offline and HfApi().model_info(config["model"]["id"],
                revision=config["model"]["revision"]).sha != config["model"]["revision"]:
            raise RuntimeError("Model revision resolution differs")
        snapshot = Path(snapshot_download(config["model"]["id"],
            revision=config["model"]["revision"], local_files_only=offline,
            allow_patterns=["*.json", "*.safetensors", "*.model", "merges.txt", "vocab.*"]))
        asset_hashes = {}
        import hashlib
        for asset in sorted(snapshot.rglob("*")):
            if asset.is_file():
                digest = hashlib.sha256()
                with asset.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                asset_hashes[asset.relative_to(snapshot).as_posix()] = {
                    "bytes": asset.stat().st_size, "sha256": digest.hexdigest()}
        common = {"trust_remote_code": False, "local_files_only": True}
        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), **common)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(snapshot), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            **common).to(self.device).eval()
        geometry = self.model.config
        if (geometry.model_type, geometry.num_hidden_layers, geometry.hidden_size) != ("qwen3", 28, 2048):
            raise RuntimeError("Pinned Qwen geometry differs")
        if any(p.dtype != torch.bfloat16 for p in self.model.parameters() if p.is_floating_point()):
            raise RuntimeError("Model parameters are not uniformly BF16")
        self.marker = list(self.tokenizer.encode("</think>", add_special_tokens=False))
        self.cue = list(self.tokenizer.encode("</think>\n\n", add_special_tokens=False))
        eos = self.model.generation_config.eos_token_id
        self.eos = [eos] if isinstance(eos, int) else list(eos)
        if self.marker != [151668] or self.cue != [151668, 271] or self.eos != [151645, 151643]:
            raise RuntimeError("Pinned control/EOS token IDs differ")
        if self.tokenizer.pad_token_id is None or self.tokenizer.eos_token_id != 151645:
            raise RuntimeError("Pinned tokenizer padding/EOS contract differs")
        generation = self.model.generation_config.to_dict()
        # A fresh artifact must record the full resolved defaults, including
        # penalties not supplied as explicit generate() kwargs.
        self.provenance = {
            "runtime_versions": observed, "model_revision": config["model"]["revision"],
            "model_snapshot_asset_hashes": asset_hashes,
            "tokenizer_revision": config["model"]["tokenizer_revision"],
            "chat_template_sha256": sha(str(self.tokenizer.chat_template).encode()),
            "model_config_sha256": sha(canonical(geometry.to_dict())),
            "resolved_generation_config": generation,
            "resolved_generation_config_sha256": sha(canonical(generation)),
            "close_marker_token_ids": self.marker, "cue_token_ids": self.cue,
            "eos_token_ids": self.eos, "pad_token_id": self.tokenizer.pad_token_id,
            "gpu_name": torch.cuda.get_device_name(self.device),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(self.device)),
            "cuda_runtime": torch.version.cuda,
            "bf16": True, "real_gpu_regeneration_exercised": True,
        }

    def prompt(self, question: str) -> dict:
        user = self.config["prompt"]["template"].replace("{problem}", question)
        messages = [{"role": "user", "content": user}]
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True,
                                                add_generation_prompt=True,
                                                enable_thinking=True)
        if not ids:
            raise RuntimeError("Chat template produced an empty prompt")
        return {"token_ids": list(map(int, ids)),
                "problem_token_count": len(self.tokenizer.encode(question, add_special_tokens=False))}

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)

    @contextmanager
    def seeded(self, seed: int | None):
        if seed is None:
            yield
            return
        torch = self.torch
        with torch.random.fork_rng(devices=[self.device.index or 0]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            yield

    def _generate(self, ids: Sequence[int], *, seed: int | None, readout: bool) -> tuple[dict, tuple]:
        torch = self.torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        cap = 64 if readout else 4096
        if not ids or len(ids) + cap > self.model.config.max_position_embeddings:
            raise RuntimeError("Frozen request exceeds model context capacity")
        kwargs = {"max_new_tokens": cap, "do_sample": not readout, "num_beams": 1,
                  "use_cache": True, "pad_token_id": self.tokenizer.pad_token_id,
                  "eos_token_id": self.eos, "return_dict_in_generate": readout,
                  "output_scores": readout, "output_hidden_states": False,
                  "output_attentions": False, "logits_to_keep": 1}
        if not readout:
            kwargs.update(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
            marker = tuple(self.marker)
            class StopClose(StoppingCriteria):
                def __call__(self, input_ids, scores, **unused):
                    return tuple(input_ids[0, -len(marker):].tolist()) == marker
            kwargs["stopping_criteria"] = StoppingCriteriaList([StopClose()])
        tensor = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        with self.seeded(seed), torch.inference_mode():
            output = self.model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor), **kwargs)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        sequences = output.sequences if readout else output
        full = list(map(int, sequences[0].detach().cpu().tolist()))
        if full[:len(ids)] != list(ids):
            raise RuntimeError("Generation did not preserve exact supplied prefix")
        generated = full[len(ids):]
        scores = tuple(output.scores) if readout else ()
        if readout and len(scores) != len(generated):
            raise RuntimeError("Readout score count differs from generated token count")
        if not readout and generated[-len(self.marker):] == self.marker:
            termination = "THINK_CLOSE_MARKER"
        elif generated and generated[-1] in self.eos:
            termination = "EOS"
        else:
            termination = "MAX_NEW_TOKENS" if len(generated) >= cap else "STOPPED_OTHER"
        receipt = {"tokens": generated, "termination": termination,
                   "elapsed_seconds": elapsed,
                   "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(self.device)}
        del tensor, output, sequences
        return receipt, scores

    def reasoning(self, ids: Sequence[int], seed: int) -> dict:
        result, _ = self._generate(ids, seed=seed, readout=False)
        self.torch.cuda.empty_cache()
        return result

    def readout(self, ids: Sequence[int]) -> dict:
        result, scores = self._generate(ids, seed=None, readout=True)
        torch = self.torch
        selected, entropy, margin = [], None, None
        try:
            for index, (token, score) in enumerate(zip(result["tokens"], scores, strict=True)):
                if tuple(score.shape)[:1] != (1,):
                    raise RuntimeError("Readout score tensor batch size differs")
                logp = torch.log_softmax(score.float(), dim=-1)
                selected.append(float(logp[0, token].item()))
                if index == 0:
                    probability = logp.exp()
                    entropy = float((-(probability * logp).sum(dim=-1)[0]).item())
                    top = torch.topk(probability[0], k=2).values
                    margin = float((top[0] - top[1]).item())
                del logp
            result["scores"] = reduce_readout(result["tokens"], selected, entropy, margin, self.decode)
            # Missing/nonfinite scalars retain missingness; no inferred answer probability.
            result["scores"], converted_fields = encode_nonfinite_scalars(result["scores"])
            result["nonfinite_scalar_fields_encoded_missing"] = converted_fields
            result["nonfinite_scalar_count_encoded_missing"] = len(converted_fields)
            return result
        finally:
            del scores
            torch.cuda.empty_cache()

    def activation(self, ids: Sequence[int]) -> tuple[bytes, dict]:
        torch = self.torch
        blocks = self.model.model.layers
        if len(blocks) != 28:
            raise RuntimeError("Transformer block count differs")
        captured, counts, handles = {}, [0] * 28, []
        def make_hook(layer):
            def hook(module, args, output):
                counts[layer] += 1
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                if counts[layer] != 1 or tuple(hidden.shape)[::2] != (1, 2048):
                    raise RuntimeError("Post-block hook geometry/coverage differs")
                captured[layer] = hidden[0, -1, :].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            return hook
        started = time.perf_counter()
        try:
            for layer, block in enumerate(blocks):
                handles.append(block.register_forward_hook(make_hook(layer)))
            tensor = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                self.model.model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                 use_cache=False, output_attentions=False,
                                 output_hidden_states=False, return_dict=False)
            torch.cuda.synchronize(self.device)
            if counts != [1] * 28:
                raise RuntimeError("Activation hooks did not execute exactly once")
            matrix = torch.stack([captured[i] for i in range(28)]).contiguous()
            if matrix.shape != (28, 2048) or not torch.isfinite(matrix).all():
                raise RuntimeError("Activation matrix geometry/finiteness differs")
            raw = matrix.view(torch.uint8).numpy().tobytes()
            header = canonical({"shape": [28, 2048], "dtype": "bfloat16",
                                "representation": "post-block-last-reasoning-token-v1"})
            stream = io.BytesIO()
            torch.save(matrix, stream)
            return stream.getvalue(), {"shape": [28, 2048], "dtype": "bfloat16",
                                      "matrix_sha256": sha(header + b"\0" + raw),
                                      "elapsed_seconds": time.perf_counter() - started}
        finally:
            for handle in handles:
                handle.remove()
            captured.clear()
            torch.cuda.empty_cache()
