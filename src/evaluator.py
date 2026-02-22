"""
Benchmarking: perplexity, latency, memory.
"""

import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset

logger = logging.getLogger(__name__)


def _device(model) -> torch.device:
    """Best-effort device detection."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def gpu_memory_mb() -> dict:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "peak_mb": 0}
    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024 ** 2,
        "peak_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
    }


def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def measure_perplexity(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_length: int = 2048,
    stride: int = 512,
    max_tokens: int = None,
) -> float:
    """
    Standard sliding-window perplexity on wikitext-2 test.

    Args:
        max_tokens: if set, only use the first N tokens (for quick runs).
    """
    logger.info("Measuring perplexity ...")
    ds = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(ds["text"])

    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    if max_tokens is not None:
        seq_len = min(seq_len, max_tokens)

    logger.info(f"  tokens={seq_len}, max_length={max_length}, stride={stride}")

    device = _device(model)
    nlls = []
    prev_end = 0

    pbar = tqdm(
        range(0, seq_len, stride),
        desc="  ppl",
        leave=False,
    )
    for begin in pbar:
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end

        ids = encodings.input_ids[:, begin:end].to(device)
        labels = ids.clone()
        labels[:, :-trg_len] = -100

        with torch.no_grad():
            loss = model(ids, labels=labels).loss

        nlls.append(loss.float().cpu())
        prev_end = end
        if end == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    logger.info(f"  perplexity = {ppl:.4f}")
    return ppl


def measure_latency(
    model,
    tokenizer,
    prompt: str = "The future of artificial intelligence is",
    n_new_tokens: int = 128,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> dict:
    """
    Measure token-generation throughput (tokens / sec).
    """
    logger.info(f"Measuring latency ({n_runs} runs, {n_new_tokens} tokens) ...")
    device = _device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inp_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=n_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    for _ in range(n_warmup):
        with torch.no_grad():
            model.generate(**inputs, **gen_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    for _ in tqdm(range(n_runs), desc="  latency", leave=False):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        gen_tok = out.shape[1] - inp_len
        latencies.append({"time_s": t1 - t0, "tokens": gen_tok})

    times = [l["time_s"] for l in latencies]
    toks = [l["tokens"] for l in latencies]
    tps = [t / s for t, s in zip(toks, times)]

    res = {
        "latency_mean_s": float(np.mean(times)),
        "latency_std_s": float(np.std(times)),
        "tokens_per_sec_mean": float(np.mean(tps)),
        "tokens_per_sec_std": float(np.std(tps)),
        "generated_tokens": int(np.mean(toks)),
    }
    logger.info(
        f"  {res['tokens_per_sec_mean']:.1f} tok/s  "
        f"(latency {res['latency_mean_s']:.3f}+/-{res['latency_std_s']:.3f}s)"
    )
    return res


def run_single_benchmark(
    model,
    tokenizer,
    method_name: str,
    *,
    ppl_max_length: int = 2048,
    ppl_stride: int = 512,
    ppl_max_tokens: int = None,
    gen_tokens: int = 128,
    latency_runs: int = 10,
    prompt: str = "The future of artificial intelligence is",
) -> dict:
    """Run all measurements for one model and return a result dict."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Benchmark: {method_name}")
    logger.info(f"{'=' * 60}")

    reset_memory_stats()
    results = {"method": method_name}

    mem = gpu_memory_mb()
    results["gpu_mem_mb"] = round(mem["allocated_mb"], 1)

    try:
        results["perplexity"] = round(
            measure_perplexity(
                model, tokenizer,
                max_length=ppl_max_length,
                stride=ppl_stride,
                max_tokens=ppl_max_tokens,
            ),
            4,
        )
    except Exception as e:
        logger.error(f"  perplexity failed: {e}")
        results["perplexity"] = None

    try:
        lat = measure_latency(
            model, tokenizer,
            prompt=prompt,
            n_new_tokens=gen_tokens,
            n_runs=latency_runs,
        )
        results["tokens_per_sec"] = round(lat["tokens_per_sec_mean"], 2)
        results["latency_s"] = round(lat["latency_mean_s"], 4)
    except Exception as e:
        logger.error(f"  latency failed: {e}")
        results["tokens_per_sec"] = None
        results["latency_s"] = None

    peak = gpu_memory_mb()
    results["peak_gpu_mb"] = round(peak["peak_mb"], 1)

    try:
        device = _device(model)
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=60, do_sample=False)
        results["sample_output"] = tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        results["sample_output"] = f"(failed: {e})"

    return results
