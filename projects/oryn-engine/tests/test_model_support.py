import pytest

from oryn_engine.model_support import transformers_workload


def test_supported_workloads_have_one_shared_engine_classification() -> None:
    assert transformers_workload({"WhisperForConditionalGeneration"}, None) == "asr"
    assert transformers_workload({"Qwen3ASRForConditionalGeneration"}, None) == "asr"
    assert (
        transformers_workload({"Qwen3ASRForTokenClassification"}, "token-classification")
        == "forced-alignment"
    )
    assert transformers_workload({"Qwen3ForCausalLM"}, None) == "causal-chat"
    assert (
        transformers_workload({"Qwen3_5ForConditionalGeneration"}, "image-text-to-text")
        == "multimodal-chat"
    )
    assert (
        transformers_workload({"MusicgenForConditionalGeneration"}, "text-to-audio") == "musicgen"
    )
    assert transformers_workload({"Qwen3ForCausalLM"}, "feature-extraction") == "embedding"
    assert transformers_workload({"Qwen3ForCausalLM"}, "text-ranking") == "reranking"


def test_pipeline_tag_alone_does_not_claim_an_unimplemented_runner() -> None:
    assert transformers_workload({"UnknownAudioModel"}, "text-to-audio") is None
    assert transformers_workload({"UnsupportedEncoder"}, "feature-extraction") is None
    assert transformers_workload({"BertModel"}, "feature-extraction") == "encoder-embedding"


@pytest.mark.parametrize(
    ("architecture", "task", "workload"),
    [
        ("DistilBertForSequenceClassification", "text-classification", "text-classification"),
        ("BertForTokenClassification", "token-classification", "token-classification"),
        ("RobertaForQuestionAnswering", "question-answering", "question-answering"),
        ("BertForMaskedLM", "fill-mask", "fill-mask"),
        ("T5ForConditionalGeneration", "summarization", "seq2seq"),
        ("ViTForImageClassification", "image-classification", "image-classification"),
        ("ASTForAudioClassification", "audio-classification", "audio-classification"),
        ("DetrForObjectDetection", "object-detection", "object-detection"),
        ("DepthAnythingForDepthEstimation", "depth-estimation", "depth-estimation"),
        ("CLIPModel", "zero-shot-image-classification", "zero-shot-image-classification"),
    ],
)
def test_breadth_workloads_are_selected_by_closed_architecture_and_task(
    architecture: str, task: str, workload: str
) -> None:
    assert transformers_workload({architecture}, task) == workload
