"""
Post-training quantization: GPTQ & AWQ.

Both quantizers:
  1. Load the original FP16 model
  2. Calibrate on a small dataset (wikitext-2 train)
  3. Quantise to INT4
  4. Save to disk
"""

import os
import logging
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _prepare_calibration_data(
    tokenizer,
    n_samples: int = 128,
    seq_len: int = 2048,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
):
    """
    Return a list of tokenised examples for GPTQ calibration.
    Each element is a dict {"input_ids": Tensor(1, L), "attention_mask": Tensor(1, L)}.
    """
    logger.info(f"Preparing calibration data ({n_samples} samples, seq_len={seq_len}) ...")
    ds = load_dataset(dataset_name, dataset_config, split="train")

    all_text = "\n\n".join(t for t in ds["text"] if t.strip())
    all_ids = tokenizer(all_text, return_tensors="pt")["input_ids"][0]

    examples = []
    for start in range(0, len(all_ids) - seq_len, seq_len):
        chunk = all_ids[start : start + seq_len]
        examples.append(
            {
                "input_ids": chunk.unsqueeze(0),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
            }
        )
        if len(examples) >= n_samples:
            break

    logger.info(f"  -> {len(examples)} calibration examples ready.")
    return examples


def quantize_gptq(
    model_name: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    n_calibration: int = 128,
    seq_len: int = 2048,
):
    """Quantise *model_name* with GPTQ and save to *output_dir*."""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    logger.info(f"GPTQ quantization: {model_name} -> {bits}-bit (group_size={group_size})")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    q_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        model_name, q_config, trust_remote_code=True,
    )

    examples = _prepare_calibration_data(tokenizer, n_calibration, seq_len)

    logger.info("Quantising (may take 15-30 min on T4) ...")
    model.quantize(examples)

    os.makedirs(output_dir, exist_ok=True)
    model.save_quantized(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"  -> saved to {output_dir}")

    return output_dir


def quantize_awq(
    model_name: str,
    output_dir: str,
    w_bit: int = 4,
    group_size: int = 128,
):
    """Quantise *model_name* with AWQ and save to *output_dir*."""
    from awq import AutoAWQForCausalLM

    logger.info(f"AWQ quantization: {model_name} -> {w_bit}-bit (group_size={group_size})")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoAWQForCausalLM.from_pretrained(model_name)

    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": w_bit,
        "version": "GEMM",
    }

    logger.info("Quantising (may take 15-30 min on T4) ...")
    model.quantize(tokenizer, quant_config=quant_config)

    os.makedirs(output_dir, exist_ok=True)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"  -> saved to {output_dir}")

    return output_dir
