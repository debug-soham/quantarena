# quantarena

An open benchmark comparing **AWQ**, **GPTQ** and **FP8** quantization on six open-weight LLMs.
For each model × method it measures:

- **perplexity drift**: WikiText-2 perplexity relative to the bf16 baseline
- **throughput**: generation tokens/s and time-to-first-token at several batch sizes (vLLM)
- **memory**: GPU memory for weights, peak memory for a full-context forward pass, size on disk

Every run writes its fully resolved config, a fingerprint and the package/GPU environment next to
its results, so any number in the table can be traced back and reproduced.

## Models

| config | checkpoint |
|---|---|
| `llama-3.1-8b` | `meta-llama/Llama-3.1-8B` |
| `mistral-7b-v0.3` | `mistralai/Mistral-7B-v0.3` |
| `qwen3-8b` | `Qwen/Qwen3-8B-Base` |
| `gemma-2-9b` | `google/gemma-2-9b` |
| `phi-4-mini` | `microsoft/Phi-4-mini-instruct` (no base model released) |
| `olmo-3-7b` | `allenai/Olmo-3-1025-7B` |

## Methods

| config | library | weights | activations | calibration |
|---|---|---|---|---|
| `bf16` | – | bf16 | bf16 | – |
| `awq-w4g128` | AutoAWQ | int4, group 128, asym | fp16 | 128 × 512 tok, pile-val |
| `gptq-w4g128` | GPTQModel | int4, group 128, sym, act-order | bf16 | 128 × 512 tok, pile-val |
| `fp8` | llm-compressor | fp8 e4m3, per-channel | fp8, dynamic per-token | none |

AWQ and GPTQ get exactly the same calibration samples (same seed, same tokenizer truncation).

## Setup

Linux + NVIDIA GPU. Native FP8 matmuls need compute capability 8.9+ (L4, L40S, 4090, H100);
on Ampere vLLM runs FP8 checkpoints with a weight-only Marlin kernel, so throughput numbers
from A100s are not comparable for that method.

```bash
uv sync --extra eval --extra awq --extra gptq --extra fp8 --extra vllm
huggingface-cli login   # llama and gemma are gated
```

The quantization libraries pin torch versions that don't always agree with each other or
with vLLM. If resolution fails, use one environment per method (`uv sync --extra eval --extra awq`, etc.).

## Usage

```bash
quantarena list                                   # available models and methods
quantarena show llama-3.1-8b awq-w4g128           # resolved config
quantarena run llama-3.1-8b awq-w4g128            # quantize + ppl + memory + throughput
quantarena run llama-3.1-8b bf16 --stages ppl     # just one stage
quantarena run qwen3-8b fp8 --set eval.seq_len=4096 --set bench.batch_sizes=[1,64]

quantarena matrix --keep-going                    # everything, one process per stage
quantarena report --md results/summary.md         # summary.csv + markdown table
```

Or `scripts/run_all.sh` for the whole sweep with logging.

Stages whose result file already exists with the same config fingerprint are skipped, and
quantized checkpoints are reused if model/quant/calib/seed are unchanged, so a crashed
sweep can simply be restarted.

## Configs

```
configs/
  base.yaml        defaults (eval, calibration, bench settings)
  models/*.yaml    one per model
  methods/*.yaml   one per quantization method
```

An experiment is `base.yaml` ← `models/<m>.yaml` ← `methods/<q>.yaml` ← `--set` overrides. Every
file has the same shape, so a method file can override model fields (AWQ sets `model.dtype: float16`)
and a model file can override eval settings. Unknown keys are errors.

## Methodology

**Perplexity.** WikiText-2 raw test split joined with `"\n\n"`, tokenized once, scored with a
sliding window (`seq_len=2048`, `stride=512`). Each token after the first is scored exactly
once, with at least `seq_len - stride` tokens of context. Loss is computed in fp32 from the
logits rather than by the model's own loss function. Drift is `ppl / ppl_bf16 - 1`.

**Memory.** Loaded through transformers on a clean CUDA context. `weights_gib` is allocated
memory after load, `forward_peak_gib` is the peak during one `seq_len` forward without KV cache.
vLLM isn't used here since it preallocates a fixed share of the GPU.

**Throughput.** vLLM offline engine, random token prompts of 512 tokens, exactly 128 output
tokens (`ignore_eos`), greedy, prefix caching disabled. 2 warmup runs, median of 5. TTFT is
measured as a separate 1-token generation for the whole batch. A transformers `generate`
backend is available (`--set bench.backend=hf`) for sanity checks but is much slower.

## Results

Each run lands in `results/<model>__<method>/`:

```
config.yaml  env.json  ppl.json  memory.json  throughput.json
```

`quantarena report` collects them into `results/summary.csv` and prints a markdown table.

## Known issues

- Gemma-2 uses eager attention (sdpa skips logit soft-capping). AWQ forces fp16, and Gemma-2
  activations can overflow in fp16, so watch its AWQ perplexity.
- Phi-4-mini is an instruct model, so its baseline perplexity isn't comparable to the others.
  Drift is still meaningful.
- AutoAWQ was archived before OLMo-3 came out and has no model class for it, so
  `olmo-3-7b` × `awq-w4g128` fails at the quantize stage (`matrix --keep-going` moves on).
- `stored_params` in memory.json counts packed int32 tensors for AWQ/GPTQ, not logical parameters.

## Tests

```bash
uv run --group dev pytest
```

The perplexity tests need torch; they're skipped otherwise.
