# LLM Flow Processes for Text-Conditioned Regression

## Installation

The code is written in Python and uses the jax framework with the equinox deep learning library.

Install [uv](https://docs.astral.sh/uv/) (Python 3.10–3.12 per `requires-python` in `pyproject.toml`). From the repository root run `uv sync` to create `.venv`, install the project in editable mode, and include the `dev` dependency group (formatting, tests). Use `source .venv/bin/activate` if you prefer an activated shell, or prefix commands with `uv run` (for example `uv run pytest tests` or `uv run task test`).

## Training the flow

`make train`, or equivalently:

```bash
RUN_DIR=outputs/flow_prior      # anywhere you like; make train RUN_DIR=... too

uv run python neural_diffusion_processes/train.py \
    --batch-size 2046 --chunk-size 250 \
    --fourier-x-features 16 --n-steps 300000 \
    --checkpoint-path "$RUN_DIR/run.eqx" \
    --plot-path "$RUN_DIR/run.png"
```

`--checkpoint-path` is a file; `config.json` is written beside it, and that
pair is what the rest of the codebase means by a checkpoint directory. Export
`NDP_FLOW_DIR=$RUN_DIR` and every other entry point - finetuning, sampling,
the diagnostics - picks it up.

`train.py` shards the batch across local devices and rejects a batch the
device count does not divide, so 2046 runs on 1, 2, 3 or 6 devices, not 4. On
96GB GH200s use 3 GPUs, or 2 with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (which
`make train` sets); one card runs out of memory. `--chunk-size` must divide
`--n-steps`.

`uv run python neural_diffusion_processes/train.py --help` documents the full option set.

## Finetuning

`make finetune`, or `scripts/finetune_mauna_loa_multi.py`, continues training the flow on four of the real NOAA series in `data/` (CO2 weekly, CH4, N2O and SF6; `data/` also carries monthly CO2, which other presets use). 
It starts from the checkpoint named by `NDP_FLOW_DIR`, overridable per run with `--base-ckpt`. No checkpoint ships with this repository, so train one first or point `NDP_FLOW_DIR` at your own.

`data/README.md` records the source URL and file format of each series and the exact window each preset slices.

## Running the method

`neural_diffusion_processes/run.py` is the entry point that applies the default settings: `DEFAULTS` pins the sample count and guidance strengths, and `sample_for_preset` / `cached_sample_for_preset` run NDP, LLM-FP and LLM-Independent against any preset in `presets.py`. The LLM-Direct baseline is `neural_diffusion_processes/llm_direct.py`. `metrics.py` scores samples with MAE, empirical CRPS and Gaussian NLPD.

Environment overrides, used consistently across the codebase:

| variable | selects |
| --- | --- |
| `NDP_FLOW_DIR` | the flow checkpoint to sample from |
| `NDP_OUTPUTS_DIR` | the LLM score caches and sample caches (default `outputs/`) |
| `NDP_N_CORR`, `NDP_TAU` | corrector steps and step scale (default 4 and 0.1) |

Rebuilding an LLM score cache from scratch needs a GPU and the Qwen weights.