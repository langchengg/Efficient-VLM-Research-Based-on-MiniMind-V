"""
Utility functions: GPU check, logging, disk size.
"""

import os
import sys
import logging
import torch


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)


def check_gpu():
    """Print GPU info. Return True if GPU is available."""
    print("=" * 50)
    print("ENVIRONMENT")
    print("=" * 50)
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version  : {torch.version.cuda}")
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_mem / 1024 ** 3
        print(f"VRAM          : {total:.1f} GB")
    print("=" * 50)
    return torch.cuda.is_available()


def get_disk_size_mb(path: str) -> float:
    """Get total size of model weight files in a directory (MB)."""
    if not os.path.isdir(path):
        return 0.0
    exts = (".safetensors", ".bin", ".pt", ".gguf")
    total = sum(
        os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path)
        if f.endswith(exts)
    )
    return total / 1024 ** 2
