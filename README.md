# LLM Quantization Benchmark

Systematic benchmarking of **post-training quantization** methods on small
Large Language Models, designed to run on a single **NVIDIA T4 GPU (Google Colab free tier)**.

## Quantization Methods Compared

| Method | Bits | Library | Approach |
|--------|------|---------|----------|
| FP16 (baseline) | 16 | transformers | No quantization |
| BnB-INT8 | 8 | bitsandbytes | LLM.int8() — mixed-precision decomposition |
| BnB-INT4 | 4 | bitsandbytes | NF4 + double quantization |
| GPTQ-INT4 | 4 | auto-gptq | One-shot weight quantization (OBQ-based) |
| AWQ-INT4 | 4 | autoawq | Activation-aware weight quantization |

## Metrics

- **Perplexity** (WikiText-2 test, sliding window) — model quality
- **Tokens/sec** (greedy generation, 128 tokens) — inference speed
- **GPU Memory** (torch.cuda.memory_allocated) — deployment cost
- **Peak GPU Memory** — worst-case memory requirement

## Quick Start (Google Colab)

```bash
# Cell 1: clone & install
!git clone https://github.com/<YOUR_USERNAME>/llm-quantization-benchmark.git
%cd llm-quantization-benchmark
!pip install -q -r requirements.txt

# Cell 2: quick run (~15 min)
!python run_benchmark.py --quick

# Cell 3: full run (~60-90 min)
!python run_benchmark.py
```

## View Results

```python
from IPython.display import display, Image
import pandas as pd

df = pd.read_csv("results/results.csv")
display(df)
display(Image("results/benchmark_results.png"))
```

## Results

*(Your benchmark tables and plots will appear in `results/` after running.)*

| Method | Perplexity↓ | Tok/s↑ | GPU Mem (MB)↓ |
|--------|-------------|--------|---------------|
| FP16 | — | — | — |
| BnB-INT8 | — | — | — |
| BnB-INT4 | — | — | — |
| GPTQ-INT4 | — | — | — |
| AWQ-INT4 | — | — | — |

![Benchmark Results](results/benchmark_results.png)

## Model

- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** (1.1B params, LLaMA architecture)

## Project Structure

```
├── src/
│   ├── model_loader.py   # Load models with 5 quant configs
│   ├── quantizer.py      # GPTQ & AWQ quantization pipelines
│   ├── evaluator.py      # Perplexity, latency, memory measurement
│   ├── report.py         # Tables, plots, report generation
│   └── utils.py          # GPU check, logging, helpers
├── run_benchmark.py      # CLI entry point
├── requirements.txt
└── results/              # Auto-generated outputs
```

## Troubleshooting

| Issue | Solution |
|-------|---------|
| `auto-gptq` install fails | `!pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/` |
| `autoawq` install fails | `!pip install autoawq --no-build-isolation` |
| GPTQ/AWQ unavailable | Use pre-quantized: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ` / `-AWQ` |
| OOM | Ensure `del model; gc.collect(); torch.cuda.empty_cache()` after each run |
| Colab disconnects | Use `--quick` flag; partial results are saved after each method |

## Key Findings

*(Fill in after running)*

1. INT4 quantization (GPTQ/AWQ) reduces GPU memory by ~60-70% with minimal perplexity degradation.
2. ...

## References

- [LLM.int8()](https://arxiv.org/abs/2208.07339) — Dettmers et al., 2022
- [GPTQ](https://arxiv.org/abs/2210.17323) — Frantar et al., 2022
- [AWQ](https://arxiv.org/abs/2306.00978) — Lin et al., 2023
- [QLoRA / NF4](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023

## License

MIT
