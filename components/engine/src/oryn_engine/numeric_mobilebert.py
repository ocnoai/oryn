"""Complete MobileBERT masked-language graph, including bottlenecks and MLM head."""


def validate_config(c):
    for key, default, lo, hi in [
        ("embedding_size", 128, 4, 2048),
        ("intra_bottleneck_size", 128, 4, 2048),
        ("type_vocab_size", 2, 1, 16),
        ("num_feedforward_networks", 4, 1, 8),
    ]:
        value = c.get(key, default)
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError("NUMERIC_MOBILEBERT_CONFIGURATION")
    true = c.get("intra_bottleneck_size", 128)
    if (
        c.get("use_bottleneck", True) is not True
        or c.get("key_query_shared_bottleneck", True) is not True
        or type(c.get("use_bottleneck_attention", False)) is not bool
        or type(c.get("trigram_input", True)) is not bool
        or c.get("normalization_type", "no_norm") not in {"no_norm", "layer_norm"}
        or c.get("position_embedding_type", "absolute") != "absolute"
        or c["hidden_size"] <= c.get("embedding_size", 128)
        or true > c["hidden_size"]
        or true % c["num_attention_heads"]
        or type(c.get("true_hidden_size", true)) is not int
        or c.get("true_hidden_size", true) != true
    ):
        raise ValueError("NUMERIC_MOBILEBERT_CONFIGURATION")


def expected_shapes(c):
    h, e, t, f, v = (
        c["hidden_size"],
        c.get("embedding_size", 128),
        c.get("intra_bottleneck_size", 128),
        c["intermediate_size"],
        c["vocab_size"],
    )
    out = {}

    def linear(name, rows, columns):
        out[name + ".weight"] = [rows, columns]
        out[name + ".bias"] = [rows]

    def norm(name, dimension):
        out[name + ".weight"] = [dimension]
        out[name + ".bias"] = [dimension]

    prefix = "mobilebert.embeddings"
    out[prefix + ".word_embeddings.weight"] = [v, e]
    out[prefix + ".position_embeddings.weight"] = [c["max_position_embeddings"], h]
    out[prefix + ".token_type_embeddings.weight"] = [c.get("type_vocab_size", 2), h]
    linear(prefix + ".embedding_transformation", h, e * (3 if c.get("trigram_input", True) else 1))
    norm(prefix + ".LayerNorm", h)
    for i in range(c["num_hidden_layers"]):
        prefix = f"mobilebert.encoder.layer.{i}"
        for name in ["query", "key", "value"]:
            linear(
                prefix + ".attention.self." + name,
                t,
                h if name == "value" and not c.get("use_bottleneck_attention", False) else t,
            )
        linear(prefix + ".attention.output.dense", t, t)
        norm(prefix + ".attention.output.LayerNorm", t)
        for suffix in [""] + [f".ffn.{j}" for j in range(c.get("num_feedforward_networks", 4) - 1)]:
            linear(prefix + suffix + ".intermediate.dense", f, t)
            linear(prefix + suffix + ".output.dense", t, f)
            norm(prefix + suffix + ".output.LayerNorm", t)
        linear(prefix + ".output.bottleneck.dense", h, t)
        norm(prefix + ".output.bottleneck.LayerNorm", h)
        for name in ["input", "attention"]:
            linear(prefix + ".bottleneck." + name + ".dense", t, h)
            norm(prefix + ".bottleneck." + name + ".LayerNorm", t)
    prefix = "cls.predictions"
    linear(prefix + ".transform.dense", h, h)
    norm(prefix + ".transform.LayerNorm", h)
    out[prefix + ".dense.weight"] = [h - e, v]
    linear(prefix + ".decoder", v, e)
    out[prefix + ".bias"] = [v]
    return out


def stable_layer_norm(value, weight, bias, epsilon):
    """Centered float32 variance avoids the MPS fused kernel's small-variance error."""
    import torch

    shifted = value - value[..., :1]
    centered = shifted - shifted.mean(dim=-1, keepdim=True)
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    return centered * torch.rsqrt(variance + epsilon) * weight + bias


def prepare_device(model, backend):
    if backend != "mps":
        return
    from torch import nn

    class StableLayerNorm(nn.LayerNorm):
        def forward(self, value):
            return stable_layer_norm(value, self.weight, self.bias, self.eps)

    for name, layer in list(model.named_modules()):
        if type(layer) is nn.LayerNorm:
            if len(layer.normalized_shape) != 1 or layer.weight is None or layer.bias is None:
                raise ValueError("NUMERIC_MOBILEBERT_NATIVE_GRAPH_CHANGED")
            replacement = StableLayerNorm(layer.normalized_shape, eps=layer.eps)
            replacement.weight, replacement.bias = layer.weight, layer.bias
            model.set_submodule(name, replacement)
