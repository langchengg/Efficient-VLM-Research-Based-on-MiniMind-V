"""
Model loading with different quantization methods.

Supported:
  - FP16 (baseline)
  - bitsandbytes INT8
  - bitsandbytes INT4 (NF4 + double quantization)
  - GPTQ INT4 (via auto-gptq)
  - AWQ  INT4 (via autoawq)
"""

import gc
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def clean_gpu():
    """Release GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_fp16(model_name: str):
    clean_gpu()
    logger.info("Loading FP16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_bnb_int8(model_name: str):
    clean_gpu()
    logger.info("Loading BnB INT8 ...")
    cfg = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_bnb_int4(model_name: str):
    clean_gpu()
    logger.info("Loading BnB INT4 (NF4) ...")
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_gptq(path: str):
    """Load an auto-gptq quantised model from *path*."""
    clean_gpu()
    logger.info(f"Loading GPTQ model from {path} ...")
    from auto_gptq import AutoGPTQForCausalLM

    model = AutoGPTQForCausalLM.from_quantized(
        path,
        device_map="auto",
        use_safetensors=True,
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_awq(path: str):
    """Load an autoawq quantised model from *path*."""
    clean_gpu()
    logger.info(f"Loading AWQ model from {path} ...")
    from awq import AutoAWQForCausalLM

    model = AutoAWQForCausalLM.from_quantized(
        path,
        fuse_layers=False,
    )
    model.eval()
    return model
