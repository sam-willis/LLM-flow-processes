.PHONY: test train finetune help

# -- Default target -------------------------------------------------
.DEFAULT_GOAL := help

# -- Tests ----------------------------------------------------------
test:
	uv run pytest tests/

# -- Training -------------------------------------------------------
# Trains the flow prior. Batch 2046 shards across 1, 2, 3 or 6 devices (not 4),
# and train.py rejects a batch that the local device count does not divide.
# On 96GB GH200s use 3 GPUs, or 2 with the 0.95 memory fraction set below; one
# card runs out of memory. Runs long (300k steps).
#
# --batch-size, --n-steps and --fourier-x-features are the CLI defaults and are
# spelled out only so the command stands on its own; --chunk-size is not, and
# must divide --n-steps.
#
# Writes the checkpoint to the --checkpoint-path FILE and config.json beside
# it.
# Where the checkpoint lands. Override per run:  make train RUN_DIR=outputs/foo
RUN_DIR ?= outputs/flow_prior

train:
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run python neural_diffusion_processes/train.py \
		--batch-size 2046 --chunk-size 250 \
		--fourier-x-features 16 --n-steps 300000 \
		--checkpoint-path $(RUN_DIR)/run.eqx \
		--plot-path $(RUN_DIR)/run.png

# -- Finetuning -----------------------------------------------------
# Finetunes the flow on the real NOAA series in data/. Set NDP_FLOW_DIR (or
# pass --base-ckpt) if the base checkpoint is not in the default location.
finetune:
	uv run python scripts/finetune_mauna_loa_multi.py

# -- Help -----------------------------------------------------------
help:
	@echo "Available targets:"
	@echo "  test             - Run pytest tests"
	@echo "  train            - Train the flow prior (needs 2-3 GPUs, long)"
	@echo "  finetune         - Finetune that checkpoint on the NOAA series in data/"
	@echo ""
	@echo "Sample caches and checkpoints are gitignored and most steps need a GPU."
	@echo "NDP_FLOW_DIR selects the checkpoint; NDP_OUTPUTS_DIR an existing cache dir."
	@echo "RUN_DIR sets where train writes (default outputs/flow_prior)."
