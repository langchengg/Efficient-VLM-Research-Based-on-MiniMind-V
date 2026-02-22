#!/usr/bin/env python3
"""
Main entry point - runs all five quantization benchmarks sequentially.

Usage (Colab or local):
    python run_benchmark.py
    python run_benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quick
"""

import argparse
import logging

from src.utils import setup_logging, check_gpu
from src.model_loader import (
    load_tokenizer, load_fp16, load_bnb_int8, load_bnb_int4,
    load_gptq, load_awq, clean_gpu,
)
from src.quantizer import quantize_gptq, quantize_awq
from src.evaluator import run_single_benchmark
from src.report import generate_full_report

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="results")
    parser.add_argument("--gptq-dir", default="models/tinyllama-gptq-4bit")
    parser.add_argument("--awq-dir", default="models/tinyllama-awq-4bit")
    parser.add_argument("--skip-gptq", action="store_true")
    parser.add_argument("--skip-awq", action="store_true")
    parser.add_argument(
        "--quick", action="store_true",
        help="Use fewer tokens / runs for a fast sanity check (~15 min total).",
    )
    args = parser.parse_args()

    setup_logging()
    check_gpu()

    tokenizer = load_tokenizer(args.model)

    bm = dict(
        ppl_max_length=2048,
        ppl_stride=512 if not args.quick else 1024,
        ppl_max_tokens=None if not args.quick else 50_000,
        gen_tokens=128 if not args.quick else 64,
        latency_runs=10 if not args.quick else 5,
    )

    all_results = []

    # 1. FP16 baseline
    logger.info("1/5  FP16 baseline")
    model = load_fp16(args.model)
    r = run_single_benchmark(model, tokenizer, "FP16", **bm)
    all_results.append(r)
    del model; clean_gpu()

    # 2. BnB INT8
    logger.info("2/5  BitsAndBytes INT8")
    model = load_bnb_int8(args.model)
    r = run_single_benchmark(model, tokenizer, "BnB-INT8", **bm)
    all_results.append(r)
    del model; clean_gpu()

    # 3. BnB INT4
    logger.info("3/5  BitsAndBytes INT4 (NF4)")
    model = load_bnb_int4(args.model)
    r = run_single_benchmark(model, tokenizer, "BnB-INT4", **bm)
    all_results.append(r)
    del model; clean_gpu()

    # 4. GPTQ INT4
    if not args.skip_gptq:
        logger.info("4/5  GPTQ INT4")
        try:
            quantize_gptq(args.model, args.gptq_dir)
            model = load_gptq(args.gptq_dir)
            r = run_single_benchmark(model, tokenizer, "GPTQ-INT4", **bm)
            all_results.append(r)
            del model; clean_gpu()
        except Exception as e:
            logger.error(f"GPTQ skipped due to error: {e}")

    # 5. AWQ INT4
    if not args.skip_awq:
        logger.info("5/5  AWQ INT4")
        try:
            quantize_awq(args.model, args.awq_dir)
            model = load_awq(args.awq_dir)
            r = run_single_benchmark(model, tokenizer, "AWQ-INT4", **bm)
            all_results.append(r)
            del model; clean_gpu()
        except Exception as e:
            logger.error(f"AWQ skipped due to error: {e}")

    generate_full_report(all_results, save_dir=args.output)
    print("\nBenchmark complete. Results in:", args.output)


if __name__ == "__main__":
    main()
