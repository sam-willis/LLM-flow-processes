"""Core types, the flow-matching transformer, samplers, conditioning and processes.

Sections appear in dependency order: custom_types, utils, models.api,
models.transformer, sampling, flow_matching, conditioning, processes -- definitions
always precede their use, which matters because several module-level bindings
(``_batch_loss``, ``_sample_n_conditional``) are built by calling into definitions
above them.
"""

from abc import ABC, abstractmethod
from math import prod
from typing import Iterator, Optional, Sequence, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from check_shapes import check_shape, check_shapes, inherit_check_shapes
from jax import vmap
from jaxtyping import UInt32


# ==========================================================================
# --- custom_types --------------------------------------------------------
# ==========================================================================

RNGKey = UInt32[jnp.ndarray, "2"]
Array = jax.Array


class XPoints(eqx.Module):
    x: Array
    mask: Array

    @check_shapes()
    def __init__(
        self,
        x: Array,
        mask: Optional[Array] = None,
    ):
        eqx.Module.__init__(self)
        mask = mask if mask is not None else jnp.zeros(x.shape[0], dtype=bool)
        self.x = check_shape(x, "[batch..., n_points, n_inputs]")
        self.mask = check_shape(mask, "[batch..., n_points]")

    def __len__(self):
        return len(self.mask)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.mask.shape

    def __getitem__(self, idx):
        return self.__class__(x=self.x[idx], mask=self.mask[idx])


class YPoints(eqx.Module):
    y: Array
    mask: Array

    @check_shapes()
    def __init__(
        self,
        y: Array,
        mask: Optional[Array] = None,
    ):
        eqx.Module.__init__(self)
        # all zeros is unmasked
        mask = mask if mask is not None else jnp.zeros(y.shape[0], dtype=bool)
        self.y = check_shape(y, "[batch..., n_points, n_outputs]")
        self.mask = check_shape(mask, "[batch..., n_points]")

    def __len__(self):
        return len(self.mask)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.mask.shape

    def __getitem__(self, idx):
        return self.__class__(y=self.y[idx], mask=self.mask[idx])


class XYPoints(XPoints, YPoints):
    def __init__(
        self,
        x: Array,
        y: Array,
        mask: Optional[Array] = None,
    ):
        # have to be careful with this dual inheritance.
        # I'm only doing this because this class is very simple
        # consider removing if this class ends up with additional functionality.
        mask = jnp.broadcast_to(mask, y.shape[:-1]) if mask is not None else None
        x = jnp.broadcast_to(x, (*y.shape[:-1], x.shape[-1]))
        XPoints.__init__(self, x, mask)
        YPoints.__init__(self, y, self.mask)

    @classmethod
    def from_points(cls, x: XPoints, y: YPoints) -> "XYPoints":
        # checkify.check(jnp.array_equal(x.mask, y.mask), "Masks of both points must be equal")
        return cls.from_points_array(x=x, y=y.y)

    @staticmethod
    def from_points_array(x: XPoints, y: Array) -> "XYPoints":
        return XYPoints(x=x.x, y=y, mask=x.mask)

    def __getitem__(self, idx):
        return self.__class__(x=self.x[idx], y=self.y[idx], mask=self.mask[idx])


# ==========================================================================
# --- utils ---------------------------------------------------------------
# ==========================================================================

JITTER = 1e-12


def get_key_iterator(seed: int) -> Iterator[RNGKey]:
    key = jax.random.PRNGKey(seed)
    while True:
        key, next_key = jax.random.split(key)
        yield next_key


def _get_stacked_key(key: RNGKey, n_keys: int) -> RNGKey:
    return jnp.stack(jax.random.split(key, n_keys))


def get_stacked_key(key: RNGKey, n_keys: int | Sequence[int]) -> RNGKey:
    """
    Get a set of key of a given shape. Useful for calling on a vmaped function.

    Not jit compilable.
    """
    if isinstance(n_keys, int):
        n_keys = [n_keys]
    size = prod(n_keys)
    return _get_stacked_key(key, size).reshape([*n_keys, -1])


@check_shapes(
    "mask: [N]",
    "where_unmasked: [N, m]",
    "where_masked: [N, m]",
    "return: [N, m]",
)
def mask_vector(
    mask: Array,
    where_unmasked: Array,
    where_masked: Array,
) -> Array:
    return jax.vmap(
        lambda _where_unmasked, _where_masked: jnp.where(mask == 0, _where_unmasked, _where_masked),
        in_axes=1,
        out_axes=1,
    )(where_unmasked, where_masked)


@check_shapes(
    "mask: [N]",
    "where_unmasked: [N, N]",
    "where_masked: [N, N]",
    "return: [N, N]",
)
def mask_matrix(
    mask: Array,
    where_unmasked: Array,
    where_masked: Array,
) -> Array:
    """
    Replace the rows and columns of a `where_unmasked` with `where_masked` where the mask is 1.

    :param mask: an N length vector of 0s and 1s, 0 is unmasked, 1 is masked.
    :param where_unmasked: any unmasked row or column will be replaced with this matrix
    :param where_masked: any unmasked row or column will be replaced with this matrix
    :return: an NxN matrix with the rows and columns corresponding to 1s in the mask replaced with the `where_masked`
    """
    _mask = mask == 0
    cov_mask = ~jnp.outer(_mask, _mask)
    return jnp.where(cov_mask, where_masked, where_unmasked)


@check_shapes(
    "delta: [n_data, output_dim]",
    "cov: [n_data, n_data]",
    "mask: [n_data]",
    "return: []",
)
def gaussian_masked_log_p(
    delta: Array,
    cov: Array,
    mask: Array,
) -> Array:
    """
    Compute the log probability of the data under the gaussian.

    :param delta: the difference between the data and the mean.
    :param cov: the covariance matrix.
    :param mask: a mask of which points are observed.
    :return: the log probability of the data under the gaussian.
    """
    _, output_dim = delta.shape
    # we're going to assume that the output dimensions are independant.
    n = (len(mask) - jnp.count_nonzero(mask)) * output_dim
    identity = jnp.eye(len(mask))

    # apply the mask to the every output dim
    delta = mask_vector(mask, delta, jnp.zeros_like(delta))
    cov = mask_matrix(mask, cov, jnp.eye(len(cov)))
    # add jitter, which will allow the matrix to be inverted, however does make the
    # calculation less accurate.
    cov = cov + JITTER * identity

    # n is for unmasked points.
    p = -0.5 * n * jnp.log(2.0 * jnp.pi)

    L = jnp.linalg.cholesky(cov)
    # masked points have unit variance, so contribute zero to log term.
    p -= jnp.sum(jnp.log(jnp.diag(L)))
    # dot product for each output
    # solve the inverse delta for each output dim
    # delta.T @ inv_cov @ delta, masked terms have 0 delta, so 0 contribution.
    inv_cov__delta = jax.vmap(
        lambda _delta: jax.scipy.linalg.solve(cov, _delta, assume_a="pos"),
        in_axes=1,
        out_axes=1,
    )(delta)
    dot = jax.vmap(jnp.dot, in_axes=(1, 1))(delta, inv_cov__delta)
    # Given the outputs are independant, sum to get the probability of the data.
    p -= 0.5 * jnp.sum(dot)
    return p


@check_shapes(
    "x: [batch...]",
    "return: []",
)
def logmeanexp(x: Array) -> Array:
    lse = jax.scipy.special.logsumexp(x)
    # this form allows extention with axis in logsumexp
    n = jnp.size(x) // jnp.size(lse)
    return jax.scipy.special.logsumexp(x) - jnp.log(n)


# ==========================================================================
# --- models.api ----------------------------------------------------------
# ==========================================================================

class Network(eqx.Module):
    @check_shapes(
        "t: []",
        "yt: [n_data, output_dim]",
        "x: [n_data, input_dim]",
        "mask: [n_data,] if mask is not None",
        "return: [n_data, output_dim]",
    )
    @abstractmethod
    def __call__(
        self,
        t: Array,
        yt: Array,
        x: Array,
        mask: Array,
        key: RNGKey | None = None,
    ) -> Array:
        ...


FlowModel: TypeAlias = Network


# ==========================================================================
# --- models.transformer --------------------------------------------------
# ==========================================================================

"""
Create a diffusion transformer-style architechture for neural flow/diffusion.

```mermaid
graph LR
    FlowMatchingTransformer --> SinusoidalPositionEmbeddings
    FlowMatchingTransformer --> EncoderBody
    EncoderBody --> TransformerBlock
    TransformerBlock --> AttentionBlock
    TransformerBlock --> FeedForwardBlock
    TransformerBlock --> AdaLN
    TransformerBlock --> RoPE
```
"""


def count_parameters(module: eqx.Module) -> int:
    """Count the number of parameters in an equinox module."""

    def is_array(x):
        return isinstance(x, jnp.ndarray)

    arrays, _ = eqx.partition(module, is_array)
    return sum(x.size for x in jax.tree_util.tree_leaves(arrays))


def print_layer_parameters(name: str, module: eqx.Module) -> int:
    """Print the number of parameters in a module and return the count."""
    param_count = count_parameters(module)
    print(f"{name}: {param_count:,} parameters")
    return param_count


def print_transformer_parameters(transformer: "FlowMatchingTransformer") -> int:
    """
    Print a detailed parameter breakdown for a FlowMatchingTransformer.

    Args:
        transformer: The FlowMatchingTransformer instance to analyze

    Returns:
        Total number of parameters in the model
    """
    print(f"\n=== FlowMatchingTransformer Parameter Breakdown ===")

    total_params = 0

    # Non-transformer components
    print("Non-Transformer Components:")
    total_params += print_layer_parameters("  Timestep Embedding", transformer.timestep_embedding)
    total_params += print_layer_parameters("  Timestep MLP 1", transformer.timestep_mlp_1)
    total_params += print_layer_parameters("  Timestep MLP 2", transformer.timestep_mlp_2)
    total_params += print_layer_parameters("  X and Y Embedding", transformer.embedding)
    total_params += print_layer_parameters("  Input Projection", transformer.input_proj)
    total_params += print_layer_parameters("  Output Projection 1", transformer.output_proj_1)
    total_params += print_layer_parameters("  Output Projection 2", transformer.output_proj_2)
    print_layer_parameters("  Dropout", transformer.dropout)  # Should be 0

    # Transformer layers - detailed breakdown per layer
    print(f"\nTransformer Layers ({transformer.n_layers} layers):")
    for i, layer in enumerate(transformer.body.layers):
        print(f"  Layer {i+1}:")

        # Break down each transformer layer's components
        attn_params = print_layer_parameters("    - Attention Block", layer.attention_block)
        ff_params = print_layer_parameters("    - Feed Forward Block", layer.ff_block)
        adaln_params = print_layer_parameters("    - AdaLN Modulation", layer.adaLN_modulation)
        norm_attn_params = print_layer_parameters("    - Layer Norm (Attention)", layer.norm_attn)
        norm_ffn_params = print_layer_parameters("    - Layer Norm (FFN)", layer.norm_ffn)

        layer_total = count_parameters(layer)
        print(f"    Layer {i+1} Total: {layer_total:,} parameters")
        total_params += layer_total
        print()  # Add blank line between layers

    print(f"\nTotal Parameters: {total_params:,}")
    print(f"=== End Parameter Breakdown ===\n")

    return total_params


class SinusoidalPositionEmbeddings(eqx.Module):
    """
    Sinusoidal positional embeddings for time variable in flow matching.

    Applies sinusoidal positional encodings to a time variable t.
    Time t is expected to be in the range [0, 1].

    These sinusoidal positional embeddings are inspired by the positional encoding
    scheme used in the Transformer architecture, particularly effective in
    capturing sequential information.

    The wavelengths are generated in a geometric progression, ranging from
    approximately 1 to `max_wavelength` across the embedding dimensions. This is
    achieved using the formula: `max_wavelength ** (2 * position / embedding_dim)`.

    Args:
        embedding_dim: The dimension of the positional embeddings. Must be an even number.
        max_wavelength: The maximum wavelength to use in the sinusoidal functions.
                        Defaults to 10000, a common value from Transformer literature.
    """

    embedding_dim: int
    max_wavelength: float = 10000.0

    def __init__(self, embedding_dim: int, max_wavelength: float = 10000.0):
        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, got {embedding_dim}")
        self.embedding_dim = embedding_dim
        self.max_wavelength = max_wavelength

    def __call__(self, t: Array) -> Array:
        """
        Applies sinusoidal positional embeddings to the input time t.

        Args:
            t: A scalar array or batched array representing time, in the range [0, 1].

        Returns:
            A JAX array of shape (..., embedding_dim) representing the
            positional encoding for the input time(s).
        """
        # 1. Create frequency bands (wavelengths) - NOTE: jax.jit probably caches these as they're constant.
        half_dim = self.embedding_dim // 2
        positions = jnp.arange(half_dim)
        # Calculate wavelengths in a geometric progression
        wavelengths = self.max_wavelength ** (2 * positions / self.embedding_dim)
        frequencies = 1 / wavelengths  # Frequencies for sine and cosine

        # 2. Calculate sinusoidal values (broadcasting happens here)
        scaled_time = t[..., None] * frequencies[None, :]  # Broadcast for batching
        sin_encodings = jnp.sin(scaled_time)
        cos_encodings = jnp.cos(scaled_time)

        # 3. Concatenante sine and cosine to get the final embedding
        embeddings = jnp.concatenate([sin_encodings, cos_encodings], axis=-1)
        return embeddings


class RoPE(eqx.Module):
    """
    Rotary Position Embedding implemented as an equinox Module.

    RoPE performs rotation of vector pairs in 2D subspaces of the embedding dimension,
    where the rotation angle is position and frequency dependent. This creates a
    natural way to encode positional information through vector rotations rather
    than additive embeddings.

    Attributes:
        dim: Dimension of the embedding (must be even)
        base: Base for the exponential in frequency computation
    """

    dim: int
    base: int
    frequencies: jnp.ndarray

    def __init__(
        self,
        dim: int,
        base: int = 10000,
    ):
        """
        Initialize the RoPE module.

        Args:
            dim: Dimension of the embedding (must be even)
            base: Base for frequency scaling
        """
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"Dimension {dim} must be even")

        self.dim = dim
        self.base = base

        # Create frequency bands: ω_i = (base^(-2i/dim)) for i = 0...d/2-1
        exponents = jnp.arange(0, dim, 2) / dim
        self.frequencies = base ** (-exponents)

    @check_shapes(
        "inputs: [seq_len, dim]",
        "positions: [seq_len]",
        "return: [seq_len, dim]",
    )
    def __call__(
        self,
        inputs: jnp.ndarray,
        positions: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Apply rotary position embeddings to the input tensor.

        Args:
            x: Input tensor of shape (seq_len, dim)
            positions: Position indices of shape (seq_len,).

        Returns:
            Tensor of same shape as input with RoPE applied
        """
        seq_length, dim = inputs.shape

        # Calculate rotation angles for each position and frequency
        # Shape: (seq_length, dim//2)
        angles = jnp.einsum("p,d->pd", positions, self.frequencies)

        # Calculate sin and cos for the rotations, then stack for rotation computation.
        # For each pair (x, y), we compute:
        # x_new = x * cos - y * sin
        # y_new = x * sin + y * cos
        # Then get arrays:
        # sin_ = [[-sin1, sin1], [-sin2, sin2], ... ]
        # cos_ = [[cos1, cos1], [cos2, cos2], ... ]
        sin = jnp.sin(angles)
        cos = jnp.cos(angles)
        sin_ = jnp.stack([-sin, sin], axis=-1)
        cos_ = jnp.stack([cos, cos], axis=-1)

        # Reshape input into pairs for rotation, rotate, then un-reshape
        # cos_ * [[x1, x2], [x3, x4], ...] + sin_ * [[x2, x1], [x4, x3], ...]  (second part is 'roll')
        inputs_reshape = inputs.reshape(seq_length, -1, 2)
        result = inputs_reshape * cos_ + jnp.roll(inputs_reshape, 1, axis=-1) * sin_
        result = result.reshape(seq_length, dim)

        return result


class AdaLN(eqx.Module):
    "Produces modulation shift and scales for adaLN-zero."

    weight: jax.Array
    bias: jax.Array

    def __init__(self, embed_dim, hidden_dim):
        # Initialised to zeros.
        self.weight = jnp.zeros((6 * hidden_dim, embed_dim))
        self.bias = jnp.zeros(6 * hidden_dim)

    def __call__(self, embedded):
        "Turns embedded [embed_dim] into a tuple of 6 scaling factors, each with size [hidden_dim]."
        linear_trans = self.weight @ embedded + self.bias
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(linear_trans, 6)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    @staticmethod
    @check_shapes(
        "x: [hidden_dim]",
        "shift: [hidden_dim]",
        "scale: [hidden_dim]",
    )
    def modulate(x, shift, scale):
        return (1 + scale) * x + shift


class AttentionBlock(eqx.Module):
    """A single transformer attention block."""

    attention: eqx.nn.MultiheadAttention
    dropout: eqx.nn.Dropout
    rope_embeddings: RoPE | None
    num_heads: int = eqx.field(static=True)
    use_rope: bool = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: RNGKey,
        use_rope: bool,
    ):
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=attention_dropout_rate,
            key=key,
        )
        self.dropout = eqx.nn.Dropout(dropout_rate)
        if use_rope:
            self.rope_embeddings = RoPE(
                dim=hidden_size // num_heads, base=10000
            )  # NOTE: default eqx.nn.MHA implies a qk_size of hidden_size // num_heads -> so we need this for RoPE
        else:
            self.rope_embeddings = None

    @check_shapes(
        "inputs: [seq_len, hidden_size]",
        "positions: [seq_len]",
        "mask: [seq_len] if mask is not None",
        "return: [seq_len, hidden_size]",
    )
    def __call__(
        self,
        inputs: Array,
        positions: Array,
        mask: Array | None,
        enable_dropout: bool = False,
        key: RNGKey | None = None,
    ) -> Array:
        if mask is not None:
            mask = self.make_self_attention_mask(mask)

        # Commented for speed
        # @check_shapes(
        #         "query_heads: [seq_len num_heads qk_size]",
        #         "key_heads: [seq_len num_heads qk_size]",
        #         "value_heads: [seq_len num_heads vo_size]",
        #         "return[0]: [seq_len num_heads qk_size]",
        #         "return[1]: [seq_len num_heads qk_size]",
        #         "return[2]: [seq_len num_heads vo_size]",
        # )
        if self.use_rope and self.rope_embeddings is not None:
            rope_embeddings = self.rope_embeddings  # Local variable for type checker

            def apply_rope(query_heads: Array, key_heads: Array, value_heads: Array) -> tuple:
                # Vmap over heads index.
                query_heads = vmap(rope_embeddings, in_axes=(1, None), out_axes=1)(
                    query_heads, 50 * positions
                )  # HACK
                key_heads = vmap(rope_embeddings, in_axes=(1, None), out_axes=1)(
                    key_heads, 50 * positions
                )  # HACK
                return query_heads, key_heads, value_heads

            process_heads_fn = apply_rope
        else:
            process_heads_fn = None

        attention_key, dropout_key = (None, None) if key is None else jax.random.split(key)

        attention_output = self.attention(
            query=inputs,
            key_=inputs,
            value=inputs,
            mask=mask,
            inference=not enable_dropout,
            key=attention_key,
            process_heads=process_heads_fn,
        )

        dropouted_attention = self.dropout(
            attention_output, inference=not enable_dropout, key=dropout_key
        )
        return dropouted_attention

    @check_shapes(
        "mask: [seq_len]",
        "return: [num_heads, seq_len, seq_len]",
    )
    def make_self_attention_mask(self, mask: Array) -> Array:
        """Create self-attention mask from sequence-level mask."""
        mask = jnp.multiply(jnp.expand_dims(mask, axis=-1), jnp.expand_dims(mask, axis=-2))
        mask = jnp.expand_dims(mask, axis=-3)
        mask = jnp.repeat(mask, repeats=self.num_heads, axis=-3)
        return mask.astype(jnp.float32)


class FeedForwardBlock(eqx.Module):
    """A single transformer feed forward block."""

    mlp: eqx.nn.Linear
    output: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_rate: float,
        key: RNGKey,
    ):
        mlp_key, output_key = jax.random.split(key)
        self.mlp = eqx.nn.Linear(
            in_features=hidden_size, out_features=intermediate_size, key=mlp_key
        )
        self.output = eqx.nn.Linear(
            in_features=intermediate_size, out_features=hidden_size, key=output_key
        )

        self.dropout = eqx.nn.Dropout(dropout_rate)

    @check_shapes("inputs: [hidden_size]", "return: [hidden_size]")
    def __call__(
        self,
        inputs: Array,
        enable_dropout: bool = True,
        key: RNGKey | None = None,
    ) -> Array:
        # NOTE: this does not take a seq_len, so no need for vmap inside.
        hidden = jax.nn.gelu(self.mlp(inputs))
        output = self.output(hidden)
        output = self.dropout(output, inference=not enable_dropout, key=key)
        return output


class TransformerBlock(eqx.Module):
    """A single transformer layer."""

    attention_block: AttentionBlock
    ff_block: FeedForwardBlock
    norm_attn: eqx.nn.LayerNorm
    norm_ffn: eqx.nn.LayerNorm
    adaLN_modulation: AdaLN

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        conditioning_embed_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: RNGKey,
        use_rope: bool,
    ):
        attention_key, ff_key = jax.random.split(key)

        self.attention_block = AttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=attention_key,
            use_rope=use_rope,
        )
        self.ff_block = FeedForwardBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            key=ff_key,
        )
        self.norm_attn = eqx.nn.LayerNorm(shape=hidden_size, use_weight=False, use_bias=False)
        self.norm_ffn = eqx.nn.LayerNorm(shape=hidden_size, use_weight=False, use_bias=False)
        self.adaLN_modulation = AdaLN(embed_dim=conditioning_embed_size, hidden_dim=hidden_size)

    @check_shapes(
        "inputs: [seq_len, hidden_size]",
        "positions: [seq_len]",
        "conditioning: [cond_embed_size]",
        "mask: [seq_len]",
        "return: [seq_len, hidden_size]",
    )
    def __call__(
        self,
        inputs: Array,
        positions: Array,
        conditioning: Array,
        mask: Array | None = None,
        *,
        enable_dropout: bool = False,
        key: RNGKey | None = None,
    ) -> Array:
        # Keys for dropout.
        seq_len = inputs.shape[0]
        attn_key, ff_key = (None, None) if key is None else jax.random.split(key)
        ff_keys = None if ff_key is None else jax.random.split(ff_key, num=seq_len)

        # adaptive Layer Norm Zero scalings (shared by all tokens).
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            conditioning
        )

        # Attention block (map over seq len)
        attn = vmap(self.norm_attn)(inputs)
        attn = vmap(AdaLN.modulate, in_axes=(0, None, None))(attn, shift_msa, scale_msa)
        attn = self.attention_block(
            attn, positions, mask, enable_dropout=enable_dropout, key=attn_key
        )
        attn = inputs + gate_msa[None, :] * attn

        # FFN (also map over seq len)
        ffn = vmap(self.norm_ffn)(attn)
        ffn = vmap(AdaLN.modulate, in_axes=(0, None, None))(ffn, shift_mlp, scale_mlp)
        ffn = jax.vmap(self.ff_block, in_axes=(0, None, 0))(attn, enable_dropout, ff_keys)
        ffn = attn + gate_mlp[None, :] * ffn

        return ffn


class EncoderBody(eqx.Module):
    """Full encoder transformer."""

    layers: list[TransformerBlock]

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        conditioning_embed_size: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: RNGKey,
        use_rope: bool,
    ):
        keys = jax.random.split(key, num=num_layers)

        # Transformer layers
        self.layers = [
            TransformerBlock(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                conditioning_embed_size=conditioning_embed_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                key=keys[i],
                use_rope=use_rope,
            )
            for i in range(num_layers)
        ]

    @check_shapes(
        "inputs: [seq_len, hidden_size]",
        "positions: [seq_len]",
        "conditioning: [cond_embed_size]",
        "mask: [seq_len] if mask is not None",
        "return: [seq_len, hidden_size]",
    )
    def __call__(
        self,
        inputs: Array,
        positions: Array,
        conditioning: Array,
        mask: Array | None = None,
        *,
        enable_dropout: bool = False,
        key: RNGKey | None = None,
    ) -> Array:
        if key is not None:
            layer_keys = jax.random.split(key, num=len(self.layers))
        else:
            layer_keys = [None] * len(self.layers)

        x = inputs  # Initialize x with inputs
        # Pass through transformer layers
        for layer, layer_key in zip(self.layers, layer_keys):
            x = layer(
                x,
                positions=positions,
                conditioning=conditioning,
                mask=mask,
                enable_dropout=enable_dropout,
                key=layer_key,
            )
        return x


class FlowMatchingTransformer(Network):
    """Transformer for flow matching."""

    body: EncoderBody
    timestep_embedding: SinusoidalPositionEmbeddings
    timestep_mlp_1: eqx.nn.Linear
    timestep_mlp_2: eqx.nn.Linear
    embedding: eqx.nn.Linear
    input_proj: eqx.nn.Linear
    output_proj_1: eqx.nn.Linear
    output_proj_2: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    n_layers: int
    fourier_x_features: int = eqx.field(static=True)
    fourier_frequencies: Array | None

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        num_heads: int,
        input_dim: int,
        output_dim: int,
        intermediate_size: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        t_embedding_dim: int,
        t_sin_features: int,
        t_mlp_hidden_dim: int,
        xy_embedding_dim: int,
        use_rope: bool,
        fourier_x_features: int = 0,
        fourier_x_base: float = 2.0,
        key: RNGKey | None = None,
    ):
        self.n_layers = n_layers
        self.fourier_x_features = fourier_x_features
        if fourier_x_features > 0:
            # Log-spaced frequencies base^0 ... base^(K-1). With base=2 and
            # to cover the Periodic kernel's plausible period range on
            # x_range=[-2, 2]. Used to expand scalar x -> [sin(omega_k x),
            # cos(omega_k x)] per input dim before concatenating with y_t.
            self.fourier_frequencies = fourier_x_base ** jnp.arange(fourier_x_features)
            x_feature_dim = input_dim * fourier_x_features * 2
        else:
            self.fourier_frequencies = None
            x_feature_dim = input_dim
        if key is None:
            key = jax.random.PRNGKey(0)  # Default key if none provided
        keys = jax.random.split(key, num=6)  # Need 6 keys now

        self.timestep_embedding = SinusoidalPositionEmbeddings(embedding_dim=t_sin_features)
        self.timestep_mlp_1 = eqx.nn.Linear(
            in_features=t_sin_features,
            out_features=t_mlp_hidden_dim,
            key=keys[1],
        )
        self.timestep_mlp_2 = eqx.nn.Linear(
            in_features=t_mlp_hidden_dim,
            out_features=t_embedding_dim,
            key=keys[2],
        )

        # Linear embedding layers for x and yt (will be followed by activation).
        # x_feature_dim accounts for optional Fourier-features expansion on x.
        self.embedding = eqx.nn.Linear(
            in_features=x_feature_dim + output_dim,
            out_features=xy_embedding_dim,
            key=keys[0],
        )

        # Input projection
        self.input_proj = eqx.nn.Linear(
            in_features=xy_embedding_dim,  # embeddings for x, yt
            out_features=hidden_dim,
            key=keys[2],
        )

        # Output projection
        self.output_proj_1 = eqx.nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, key=keys[3]
        )
        output_proj_2 = eqx.nn.Linear(
            in_features=hidden_dim, out_features=1, key=keys[4], use_bias=True
        )
        self.output_proj_2 = eqx.tree_at(
            lambda x: (x.weight, x.bias),
            output_proj_2,
            (jnp.zeros_like(output_proj_2.weight), jnp.zeros_like(output_proj_2.bias)),
        )

        # Dropout
        self.dropout = eqx.nn.Dropout(dropout_rate)

        # Body
        self.body = EncoderBody(
            hidden_size=hidden_dim,  # Use hidden_dim here
            intermediate_size=intermediate_size,
            conditioning_embed_size=t_embedding_dim,
            num_layers=n_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=keys[5],
            use_rope=use_rope,
        )

        # Print parameter counts per layer
        print_transformer_parameters(self)

    @inherit_check_shapes
    def __call__(
        self,
        t: Array,
        yt: Array,
        x: Array,
        mask: Array,
        key: RNGKey | None = None,
        enable_dropout: bool = False,
    ) -> Array:
        """
        Args:
            t: Time input, shape [].
            yt: Data input yt, shape [seq_len, input_dim].
            x: Conditioning input x, shape [seq_len, input_dim].
            mask: Mask for data and condition, shape [seq_len].
            key: Random key for dropout.
            enable_dropout: Whether to enable dropout.

        Returns:
            Output array of shape [seq_len, output_dim].
        """
        if enable_dropout and key is None:
            raise ValueError("Key must be provided when dropout is enabled")
        body_key, dropout_key = jax.random.split(key) if key is not None else (None, None)

        t_embedding = self.timestep_embedding(t).squeeze(
            0
        )  # Here get rid of extra 1 dim (since embedding adds a batch dim)
        t_embedding = self.timestep_mlp_1(t_embedding)  # same t for all tokens
        t_embedding = jax.nn.gelu(t_embedding)
        t_embedding = self.timestep_mlp_2(t_embedding)

        if self.fourier_frequencies is not None:
            # x: [seq_len, input_dim]. Broadcast-multiply by frequencies to get
            # [seq_len, input_dim, K], apply sin/cos, concatenate -> [seq_len, input_dim*2K]
            angles = x[..., None] * self.fourier_frequencies
            x_expanded = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
            x_expanded = x_expanded.reshape(x.shape[0], -1)
            inputs = jnp.concatenate([x_expanded, yt], axis=-1)
        else:
            inputs = jnp.concatenate([x, yt], axis=-1)
        inputs = vmap(self.embedding)(inputs)
        inputs = jax.nn.gelu(inputs)
        inputs = vmap(self.input_proj)(inputs)

        # Pass through encoder layers.
        output = self.body(
            inputs,
            positions=x.squeeze(-1),  # used only for RoPE TODO: assumes 1D xs if needed
            conditioning=t_embedding,
            mask=mask,
            enable_dropout=enable_dropout,
            key=body_key,
        )

        # Dropout, output MLP+projection
        output = self.dropout(output, inference=not enable_dropout, key=dropout_key)
        output = vmap(self.output_proj_1)(output)
        output = output / jnp.sqrt(
            float(self.n_layers)
        )  # normalise by sqrt(n_layers)
        output = jax.nn.gelu(output)
        output = vmap(self.output_proj_2)(output)

        return output


# ==========================================================================
# --- sampling ------------------------------------------------------------
# ==========================================================================

# eqx does not play nice with ABCs, don't use them.
class BaseSampler(eqx.Module):
    n_steps: int
    num_corrections: int
    tau: float

    @abstractmethod
    def get_times(self) -> tuple[int, Array, float]:
        pass

    @abstractmethod
    def score(self, model: Network, t: Array, x: XPoints, y: Array) -> Array:
        pass

    @abstractmethod
    def next_y(self, model: Network, t: Array, x: XPoints, y: Array, dt: float) -> Array:
        pass


class VelocitySampler(BaseSampler):
    start_time: float = 0.1
    n_steps: int
    num_corrections: int
    tau: float

    def get_times(self):
        ts = jnp.linspace(self.start_time, 1.0, self.n_steps + 1)
        dt = 1.0 / self.n_steps
        return self.n_steps, ts, dt

    def score(self, model: FlowModel, t: Array, x: XPoints, y: Array):
        v = model(t, y, x.x, x.mask)

        # For Gaussian OT path, eq 4.78 FM:
        # t v = x + ((1-t)t + (1-t)^2) score = x + (1-t) * score
        return (t * v - y) / (1 - t)  # TODO: this diverges...

    def next_y(self, model: FlowModel, t: Array, x: XPoints, y: Array, dt: float):
        v = model(t, y, x.x, x.mask)
        return y + v * dt


@check_shapes(
    "x.x: [n_data, d_input]",
    "x.mask: [n_data]",
    "return[0].y: [n_data, d_output]",
    "return[1]: [n_steps, n_data, d_output]",
)
@eqx.filter_jit
def sample(
    model: Network,
    sampler: BaseSampler,
    key: RNGKey,
    x: XPoints,
) -> tuple[YPoints, Array]:
    y_shape = (len(x), 1)  # TODO: note this is hardcoding the d_output=1 for now
    y_init = jax.random.normal(key, y_shape)
    n_steps, ts, dt = sampler.get_times()

    def corrector_step(state: tuple[Array, RNGKey], t) -> tuple[tuple[Array, RNGKey], None]:
        """
        langevin monte carlo steps, depending on model score.
        """
        y, key = state
        noise_key, next_key = jax.random.split(key)
        eps = jax.random.normal(noise_key, y.shape)
        score = sampler.score(model, t, x, y)

        # update the step size in relation to the score
        # TODO: copied from rozet et al... various different motivations exist for different step sizes schedules here.
        delta = sampler.tau / jnp.mean(score**2)
        # LMC step: x(t) + δs + √(2δ)ϵ
        y = y + delta * score + jnp.sqrt(2 * delta) * eps
        return (y, next_key), None

    def time_step(state: tuple[Array, RNGKey], t: Array) -> tuple[tuple[Array, RNGKey], Array]:
        """
        timestep plus langevin corrections, using next_y from model
        """
        y, key = state
        y_next = sampler.next_y(model, t, x, y, dt)
        (y_next_corrected, next_key), _ = jax.lax.scan(
            corrector_step,
            (y_next, key),
            jnp.repeat(t + dt, sampler.num_corrections),
        )
        y_next_maybe_corrected = jax.lax.select(t > 0.90, y_next, y_next_corrected)
        return (y_next_maybe_corrected, next_key), y_next_maybe_corrected

    (y_out, _), ys_trajectory = jax.lax.scan(
        f=time_step,
        init=(y_init, key),
        xs=ts[:-1],
        length=n_steps,
    )
    return YPoints(y_out, x.mask), ys_trajectory


# ==========================================================================
# --- flow_matching -------------------------------------------------------
# ==========================================================================

# TODO: for now, both xs and ys are 1D only (???)
@check_shapes(
    "data_points.x: [n_input, 1]",
    "data_points.y: [n_input, 1]",
    "data_points.mask: [n_input]",
    "t: []",
    "return: []",
)
def flow_matching_single_loss(
    flow_model: FlowModel,
    data_points: XYPoints,
    t: Array,
    key: RNGKey,
) -> Array:
    # Get xs for data, corresponding y_0s, and generate latent y_1s.
    x = data_points.x
    y_1 = data_points.y
    y_0 = jax.random.normal(shape=y_1.shape, key=key)
    y_t = (1 - t) * y_0 + t * y_1
    dy_t = y_1 - y_0
    predicted_flow = flow_model(t, y_t, x, data_points.mask)
    mean_square_err = jnp.mean((predicted_flow - dy_t) ** 2)
    return mean_square_err


# Vmap over xs, ys, times and keys
_batch_loss = eqx.filter_vmap(flow_matching_single_loss, in_axes=(None, 0, 0, 0))


def _sample_times_uniform_grid(key: RNGKey, batch_size: int, dtype) -> Array:
    """Current method: low-discrepancy sampling with small random offset"""
    dt = jax.random.uniform(key, dtype=dtype, maxval=1 / batch_size)
    t = jnp.arange(batch_size, dtype=dtype) / batch_size
    return t + dt


def _sample_times_uniform_random(key: RNGKey, batch_size: int, dtype) -> Array:
    """True uniform sampling over [0,1)"""
    return jax.random.uniform(key, shape=(batch_size,), dtype=dtype, minval=0.0, maxval=1.0)


def _sample_times_logit_normal(key: RNGKey, batch_size: int, dtype) -> Array:
    """Logit-normal sampling (SD-3 method) - better coverage of boundaries"""
    z = jax.random.normal(key, shape=(batch_size,))
    return jax.nn.sigmoid(z).astype(dtype)


@check_shapes(
    "data_points.x: [batch_size, n_input, 1]",
    "data_points.y: [batch_size, n_input, 1]",
    "data_points.mask: [batch_size, n_input]",
    "return: []",
)
def batch_loss(
    flow_model: FlowModel,
    data_points: XYPoints,
    key: RNGKey,
    time_sampling_strategy: str = "uniform_grid",
):
    """
    Loss function for the diffusion process.

    Args:
        flow_model: The flow model to train
        data_points: Batch of data points
        key: Random key for sampling
        time_sampling_strategy: Strategy for sampling time points
            - "uniform_grid": Current method with low-discrepancy sampling
            - "uniform_random": True uniform sampling over [0,1)
            - "logit_normal": Logit-normal sampling (SD-3 method)
    """
    batch_size = len(data_points)

    # Select sampling function based on strategy (outside JAX compilation)
    if time_sampling_strategy == "uniform_grid":
        times = _sample_times_uniform_grid(key, batch_size, data_points.y.dtype)
    elif time_sampling_strategy == "uniform_random":
        times = _sample_times_uniform_random(key, batch_size, data_points.y.dtype)
    elif time_sampling_strategy == "logit_normal":
        times = _sample_times_logit_normal(key, batch_size, data_points.y.dtype)
    else:
        # Default to uniform_grid
        times = _sample_times_uniform_grid(key, batch_size, data_points.y.dtype)

    return jnp.mean(_batch_loss(flow_model, data_points, times, get_stacked_key(key, batch_size)))


# ==========================================================================
# --- conditioning --------------------------------------------------------
# ==========================================================================

DEFAULT_GAMMA = 1e0

### Likelihood Transformations ###


# eqx does not play nice with ABCs, don't use them.
class ConditioningHandler(eqx.Module):
    """Handler for managing context and target points. Call returns the additional log prob term."""

    target: XYPoints | XPoints
    context: XYPoints

    @property
    def combined_mask(self) -> Array:
        return jnp.concatenate([self.context.mask, self.target.mask])

    @property
    def combined_x(self) -> XPoints:
        return XPoints(jnp.concatenate([self.context.x, self.target.x]), self.combined_mask)


    @property
    def n_context(self) -> int:
        return len(self.context)

    @property
    def n_target(self) -> int:
        return len(self.target)

    def get_context(self, points: YPoints) -> YPoints:
        """Extract context points from the given YPoints."""
        return YPoints(
            y=points.y[: self.n_context],
            mask=self.combined_mask[: self.n_context],
        )

    def get_target(self, points: YPoints) -> YPoints:
        """Extract target points from the given YPoints."""
        return YPoints(
            y=points.y[self.n_context :],
            mask=self.combined_mask[self.n_context :],
        )

    def split_points(self, array: Array) -> tuple[Array, Array]:
        """Split an array into context and target components."""
        return array[: self.n_context], array[self.n_context :]

    @abstractmethod
    @check_shapes(
        "pred_output: [n_data, n_output]",
        "pred_output_var: []",
        "return: []",
    )
    def __call__(self, pred_output: Array, pred_output_var: Array) -> Array:
        pass


class ConditioningX(ConditioningHandler):
    """Conditioning handler for target points that are XPoints (X only)."""

    target: XPoints
    context: XYPoints
    likelihood_variance: float = 0.01
    gamma_scaling_ndp: float = DEFAULT_GAMMA

    @inherit_check_shapes
    def __call__(self, pred_output: Array, pred_output_var: Array) -> Array:
        """Conditional log prob integrated over prob_xys."""
        target_y1 = self.context.y
        context_pred_y1, _ = self.split_points(pred_output)
        delta = target_y1 - context_pred_y1
        var = (self.likelihood_variance + pred_output_var) * jnp.eye(len(delta))
        return self.gamma_scaling_ndp * gaussian_masked_log_p(delta, var, self.context.mask)


class ConditioningDistributionalConvolved1D(ConditioningHandler):
    """Conditional log prob handler for target points that are XYPoints (X and Y)."""

    target: XPoints
    context: XYPoints
    y_grid: Array
    prob_xys: Array
    likelihood_variance: float = 0.01  # HACK: this is a hack to stop blowing up near t=1
    gamma_scaling_distributional: float = DEFAULT_GAMMA

    @check_shapes(
        "target: [n_target]",
        "context: [n_context]",
        "y_grid: [1, n_y_grid]",
        "prob_xys: [n_target, n_y_grid]",
        "return: []",
    )
    def __init__(
        self,
        target: XPoints,
        context: XYPoints,
        y_grid: Array,
        prob_xys: Array,
        likelihood_variance: float = 0.01,
        gamma_scaling_distributional: float = DEFAULT_GAMMA,
    ):
        super().__init__(target, context)
        self.y_grid = y_grid.squeeze()
        self.prob_xys = prob_xys
        self.likelihood_variance = likelihood_variance
        self.gamma_scaling_distributional = gamma_scaling_distributional

    @check_shapes(
        "pred_output: [n_data, 1]",
        "pred_output_var: []",
        "return: []",
    )
    def __call__(self, pred_output: Array, pred_output_var: Array) -> Array:
        """Conditional log prob integrated over prob_xys."""
        # Only consider the target points for the distributional matching:
        n_target = len(self.target)
        pred_output_target = pred_output[-n_target:, 0]
        bin_width = self.y_grid[1] - self.y_grid[0]
        bin_edges = jnp.concatenate(
            [self.y_grid, jnp.array([self.y_grid[-1] + bin_width])]
        )  # [n_y_grid + 1]
        cdf = jax.scipy.stats.norm.cdf(
            (bin_edges[None, :] - pred_output_target[:, None]) / jnp.sqrt(pred_output_var), 0, 1
        )  # [n_target, n_y_grid + 1]
        gaussian_term = cdf[:, 1:] - cdf[:, :-1]  # [n_target, n_y_grid]
        q_tilde = jnp.sum(gaussian_term * self.prob_xys, axis=-1)  # sum over the y_grid bins
        return self.gamma_scaling_distributional * jnp.sum(
            jnp.log(q_tilde)
        )  # sum over the target points


class ConditioningDistributionalConvolved1DAndX(ConditioningHandler):
    target: XPoints
    context: XYPoints
    distributional_handler: ConditioningDistributionalConvolved1D = eqx.field(default=None)
    x_handler: ConditioningX = eqx.field(default=None)

    @check_shapes(
        "target: [n_target]",
        "context: [n_context]",
        "y_grid: [1, n_y_grid]",
        "prob_xys: [n_target, n_y_grid]",
        "return: []",
    )
    def __init__(
        self,
        target: XPoints,
        context: XYPoints,
        y_grid: Array,
        prob_xys: Array,
        likelihood_variance: float = 0.01,
        gamma_scaling_distributional: float = DEFAULT_GAMMA,
        gamma_scaling_ndp: float = DEFAULT_GAMMA,
    ):
        super().__init__(target, context)
        self.distributional_handler = ConditioningDistributionalConvolved1D(
            target, context, y_grid, prob_xys, likelihood_variance, gamma_scaling_distributional
        )
        self.x_handler = ConditioningX(target, context, likelihood_variance, gamma_scaling_ndp)

    @inherit_check_shapes
    def __call__(self, pred_output: Array, pred_output_var: Array) -> Array:
        """Conditional log prob integrated over prob_xys."""
        log_prob_distributional = self.distributional_handler(pred_output, pred_output_var)
        log_prob_x = self.x_handler(pred_output, pred_output_var)
        return log_prob_distributional + log_prob_x


### Conditional Models ###


class ConditionalFlow(FlowModel):
    "Take a flow model and wrap it using a conditioning handler."

    flow_model: FlowModel
    conditioning_handler: ConditioningHandler

    @inherit_check_shapes
    def __call__(self, t: Array, yt: Array, x: Array, mask: Array) -> Array:
        sigma_t = 1 - t
        alpha_t = t
        rt2 = (
            sigma_t**2 / (sigma_t**2 + alpha_t**2) + 0.01
        )  # HACK: this is a hack to stop blowing up near t=1

        def conditional_log_prob(yt: Array):
            "Conditional log prob integrated over prob_xys."
            vt = self.flow_model(t, yt, x, mask)
            y1_hat = yt + vt * (1 - t)  # [n_data, n_output]
            return self.conditioning_handler(y1_hat, rt2), vt

        cs, vt = jax.grad(conditional_log_prob, has_aux=True)(yt)
        vt_aug = vt + (1 - t) / t * cs  # NOTE: gamma is included in the conditioning handler
        return jax.lax.select(t < 0.05, vt, vt_aug)


### Helper functions for conditional sampling ###


def _sample_conditional(
    key: RNGKey,
    conditional_model: Network,
    handler: ConditioningHandler,
    sampler: BaseSampler,
) -> tuple[XYPoints, XYPoints, Array]:
    y_final, ys_trajectory = sample(conditional_model, sampler, key, handler.combined_x)
    target_predictions = XYPoints.from_points(handler.target, handler.get_target(y_final))
    context_predictions = XYPoints.from_points(handler.context, handler.get_context(y_final))
    return target_predictions, context_predictions, ys_trajectory


_sample_n_conditional = eqx.filter_jit(
    eqx.filter_vmap(
        _sample_conditional,
        in_axes=(0, None, None, None),
    )
)


def sample_n_conditional(
    n_samples: int,
    key: RNGKey,
    conditional_model: Network,
    handler: ConditioningHandler,
    sampler: BaseSampler,
    chunk_size: int | None = None,
):
    """Generate n_samples with trajectories from the conditional model.

    `chunk_size` splits the vmap into batches of at most that many samples,
    trading a little launch overhead for a proportional cut in peak device
    memory. Results are unchanged: `get_stacked_key` derives one key per
    sample up front and slicing that array hands every sample the same key it
    would have received from a single full-width vmap, so chunked and
    unchunked runs agree sample for sample.

    Needed for handlers whose per-step intermediates scale with a y-grid, e.g.
    `ConditioningDistributionalConvolved1D` materialises a
    `[n_target, n_y_grid + 1]` Gaussian CDF per sample per solver step. At 800
    samples and a 2515-bin grid that reaches ~64 GiB and OOMs an 80 GB card.
    """
    keys = get_stacked_key(key, n_samples)
    if chunk_size is None or chunk_size >= n_samples:
        return _sample_n_conditional(keys, conditional_model, handler, sampler)

    outs = [
        _sample_n_conditional(
            keys[i : i + chunk_size], conditional_model, handler, sampler
        )
        for i in range(0, n_samples, chunk_size)
    ]

    def _cat(*parts):
        if all(eqx.is_array(p) for p in parts):
            return jnp.concatenate(parts, axis=0)
        return parts[0]

    return jax.tree_util.tree_map(_cat, *outs)


# ==========================================================================
# --- processes -----------------------------------------------------------
# ==========================================================================

DEFAULT_STEPS = 100
DEFAULT_CORRECTIONS = 4
DEFAULT_TAU = 0.1


class Process(ABC):
    """Abstract base class for different diffusion methods."""

    def __init__(self, **kwargs):
        super().__init__()
        self._x_range = kwargs.get("x_range", None)
        self._y_range = kwargs.get("y_range", None)
        if self._x_range is None or self._y_range is None:
            raise ValueError("x_range and y_range must be provided")

    def normalise_x_value(self, x_value: Array) -> Array:
        """
        Normalise x value to [-2, 2] using the x_range.
        """
        x_range_delta = self._x_range[1] - self._x_range[0]
        zero_one_norm = (x_value - self._x_range[0]) / x_range_delta
        minus_two_two_norm = (zero_one_norm - 0.5) * 4
        return minus_two_two_norm

    def denormalise_x_value(self, x_value: Array) -> Array:
        """
        Denormalise x value to the original range using the x_range.
        """
        zero_one_norm = x_value / 4 + 0.5
        x_range_delta = self._x_range[1] - self._x_range[0]
        return zero_one_norm * x_range_delta + self._x_range[0]

    def normalise_y_value(self, y_value: Array) -> Array:
        """
        Normalise y value to [-1, 1] using the y_range.
        """
        y_range_delta = self._y_range[1] - self._y_range[0]
        zero_one_norm = (y_value - self._y_range[0]) / y_range_delta
        minus_one_one_norm = (zero_one_norm - 0.5) * 2
        return minus_one_one_norm

    def denormalise_y_value(self, y_value: Array) -> Array:
        """
        Denormalise y value to the original range using the y_range.
        """
        zero_one_norm = y_value / 2 + 0.5
        y_range_delta = self._y_range[1] - self._y_range[0]
        return zero_one_norm * y_range_delta + self._y_range[0]

    def normalise_xs(self, xs: XPoints) -> XPoints:
        return XPoints(self.normalise_x_value(xs.x), xs.mask)


    def denormalise_ys(self, ys: YPoints) -> YPoints:
        return YPoints(self.denormalise_y_value(ys.y), ys.mask)

    def normalise_xys(self, xys: XYPoints) -> XYPoints:
        return XYPoints(self.normalise_x_value(xys.x), self.normalise_y_value(xys.y), xys.mask)

    def denormalise_xys(self, xys: XYPoints) -> XYPoints:
        return XYPoints(self.denormalise_x_value(xys.x), self.denormalise_y_value(xys.y), xys.mask)

    @abstractmethod
    def sample(self, model: Network, target: XPoints, key: RNGKey) -> YPoints:
        """Generate unconditional samples."""
        pass

    @abstractmethod
    def sample_n_conditionals(
        self,
        model: Network,
        target: XPoints,
        context: XYPoints,
        key: RNGKey,
        n_samples: int,
        language_context: str = "",
        llm_nll_extra: dict | None = None,
        **kwargs,
    ) -> tuple[XYPoints, XYPoints, Array | None]:
        pass


class FlowProcess(Process):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gamma_scaling_ndp = kwargs.get("gamma_scaling_ndp", DEFAULT_GAMMA)

    def _get_sampler(self) -> VelocitySampler:
        return VelocitySampler(
            n_steps=DEFAULT_STEPS, num_corrections=DEFAULT_CORRECTIONS, tau=DEFAULT_TAU
        )

    def sample(self, model: FlowModel, target: XPoints, key: RNGKey) -> tuple[YPoints, Array]:
        sampler = self._get_sampler()
        normalised_target = self.normalise_xs(target)
        normalised_target, trajectory = sample(model, sampler, key, normalised_target)
        return self.denormalise_ys(normalised_target), self.denormalise_y_value(trajectory)

    def sample_n_conditionals(
        self,
        model: FlowModel,
        target: XPoints,
        context: XYPoints,
        key: RNGKey,
        n_samples: int,
        likelihood_noise_std: float = 0.0,
        language_context: str = "",
        llm_nll_extra: dict | None = None,
        chunk_size: int | None = None,
    ) -> tuple[XYPoints, XYPoints, Array]:
        del llm_nll_extra
        sampler = self._get_sampler()
        normalised_target = self.normalise_xs(target)
        normalised_context = self.normalise_xys(context)
        conditioning_handler = ConditioningX(
            normalised_target,
            normalised_context,
            likelihood_variance=likelihood_noise_std**2,
            gamma_scaling_ndp=self.gamma_scaling_ndp,
        )
        conditional_model = ConditionalFlow(model, conditioning_handler)
        normalised_target, normalised_context, trajectory = sample_n_conditional(
            n_samples, key, conditional_model, conditioning_handler, sampler,
            chunk_size=chunk_size,
        )
        return (
            self.denormalise_xys(normalised_target),
            self.denormalise_xys(normalised_context),
            self.denormalise_y_value(trajectory),
        )


