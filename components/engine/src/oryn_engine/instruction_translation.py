"""Reviewed instruction-translation bindings, selected by native graph identity."""

from oryn_engine import hunyuan_translation, qwen_translation

ARCHITECTURES = {**qwen_translation.ARCHITECTURES, **hunyuan_translation.ARCHITECTURES}


def supports(config, task):
    return task == "translation" and config.get("model_type") in ARCHITECTURES


def adapter(config):
    if qwen_translation.supports(config, "translation"):
        return qwen_translation
    if hunyuan_translation.supports(config, "translation"):
        return hunyuan_translation
    raise ValueError("TRANSLATION_NATIVE_BINDING_UNAVAILABLE")
