"""Dataclass configs describing a model, and the loader that rebuilds one.

`load_model` is the only entry point that matters: it takes a
`DiffusionModelConfig` and a checkpoint path, builds the network from
`noise_model`, and deserialises the weights into it. `flow.py` assembles the
config; everything else here is the surrounding type hierarchy.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import equinox as eqx
from omegaconf import MISSING


from neural_diffusion_processes.core import RNGKey
from neural_diffusion_processes.core import Network
from neural_diffusion_processes.core import FlowMatchingTransformer
from neural_diffusion_processes.core import Process
from neural_diffusion_processes.core import get_key_iterator


@dataclass
class DiffusionConfig:
    """Base of the config hierarchy.

    Carries no behaviour: `load_model` reads only `noise_model`. It exists
    as the declared type of `DiffusionModelConfig.diffusion`.
    """


@dataclass
class FlowConfig(DiffusionConfig):
    """Flow-matching diffusion config.

    Has no fields. `batch_loss` takes the time-sampling strategy as a string
    argument and defaults to "uniform_grid"; nothing in this repo passes
    anything else.
    """


@dataclass
class LLMFlowConfig2(FlowConfig):
    condition_ndp_part: bool = False
    gamma_scaling_distributional: float = MISSING
    gamma_scaling_ndp: float = 0.0
    # None = `prompts.DEFAULT_LLM_PATH` (Qwen/Qwen3.6-27B). Not read by `load_model`;
    # the LLM scoring code takes the model path as its own argument.
    llm_path: Optional[str] = None
    llm_type: Optional[str] = None

    def __post_init__(self):
        if self.condition_ndp_part and self.gamma_scaling_ndp == 0.0:
            logging.warning(
                "gamma_scaling_ndp is 0.0, which means the NDP part will not be conditioned."
            )


@dataclass
class NetworkConfig(ABC):
    @abstractmethod
    def create_network(self, key: RNGKey, diffusion_method: Optional[Process] = None) -> Network:
        pass


@dataclass
class BaseNeuralNetworkConfig(NetworkConfig):
    """Base config class for neural networks with common parameters."""

    num_layers: int = MISSING
    hidden_dim: int = MISSING
    num_heads: int = MISSING


@dataclass
class TransformerConfig(BaseNeuralNetworkConfig):
    """Configuration for Flow Matching Transformer."""

    name: str = "transformer"
    use_rope: bool = False
    input_dim: int = 1
    output_dim: int = 1
    intermediate_size: int = 64
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.1
    t_sin_features: int = 16
    t_mlp_hidden_dim: int = 16
    t_embedding_dim: int = 16
    xy_embedding_dim: int = 16
    # 0 = off (raw scalar x passed through linear embedding). K>0 expands
    # each x dim into [sin(base^i x), cos(base^i x)] for i=0..K-1 before
    # concatenation with y_t, giving the network explicit frequency bases.
    fourier_x_features: int = 0
    fourier_x_base: float = 2.0

    def create_network(self, key: RNGKey, diffusion_method: Optional[Process] = None) -> Network:
        return FlowMatchingTransformer(
            n_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            intermediate_size=self.intermediate_size,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            t_embedding_dim=self.t_embedding_dim,
            t_sin_features=self.t_sin_features,
            t_mlp_hidden_dim=self.t_mlp_hidden_dim,
            xy_embedding_dim=self.xy_embedding_dim,
            use_rope=self.use_rope,
            fourier_x_features=self.fourier_x_features,
            fourier_x_base=self.fourier_x_base,
            key=key,
        )


@dataclass
class DiffusionModelConfig:
    diffusion: DiffusionConfig = MISSING
    noise_model: NetworkConfig = MISSING


def load_model(model_config: DiffusionModelConfig, model_path: str) -> Network:
    """Load a model from a saved file."""
    # Create a dummy key for model creation
    key = next(get_key_iterator(42))

    # Create the network using the config
    network = model_config.noise_model.create_network(key)

    try:
        loaded_network = eqx.tree_deserialise_leaves(model_path, network)
    except Exception as e:
        raise RuntimeError(
            f"Weights in {model_path} do not match the network built from the "
            "`noise_model` config (architecture or layer sizes differ). When loading "
            "via `flow.load_ndp_flow_network_local`, that config comes from the "
            "config.json beside the weights: check it is the one `train.py` wrote "
            "for this run.\n"
            f"Original error: {e}"
        ) from e

    return loaded_network
