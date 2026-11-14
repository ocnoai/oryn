from types import SimpleNamespace

import torch

from oryn_engine.runners import _normalize_transformers_task_output


def test_question_answer_must_come_from_context_with_exact_character_offsets():
    # The question has the largest logit. It must never become the answer.
    output = SimpleNamespace(
        start_logits=torch.tensor([[0.0, 100.0, 0.0, 5.0, 1.0]]),
        end_logits=torch.tensor([[0.0, 100.0, 0.0, 5.0, 1.0]]),
    )
    value = {"context": "Paris is nice.", "maxAnswerLength": 1}
    result = _normalize_transformers_task_output(
        "question-answering",
        value,
        output,
        None,
        None,
        {},
        None,
        {
            "sequence_ids": [None, 0, None, 1, 1],
            "offsets": [[0, 0], [0, 4], [0, 0], [0, 5], [6, 8]],
        },
    )
    assert result["answer"] == "Paris"
    assert (result["start"], result["end"]) == (0, 5)


def test_multilabel_classification_uses_independent_probabilities():
    model = SimpleNamespace(
        config=SimpleNamespace(problem_type="multi_label_classification", id2label={})
    )
    result = _normalize_transformers_task_output(
        "text-classification",
        {"topK": 3},
        SimpleNamespace(logits=torch.zeros(1, 3)),
        None,
        model,
        {},
        None,
    )
    assert [row["score"] for row in result["predictions"]] == [0.5, 0.5, 0.5]


def test_token_labels_exclude_special_tokens_and_keep_text_offsets():
    processor = SimpleNamespace(
        all_special_ids=[0, 2], convert_ids_to_tokens=lambda _ids: ["[CLS]", "Paris", "[SEP]"]
    )
    model = SimpleNamespace(config=SimpleNamespace(id2label={1: "LOCATION"}))
    result = _normalize_transformers_task_output(
        "token-classification",
        {},
        SimpleNamespace(logits=torch.tensor([[[2.0, 0.0], [0.0, 4.0], [2.0, 0.0]]])),
        processor,
        model,
        {"input_ids": torch.tensor([[0, 1, 2]])},
        None,
        {"offsets": [[0, 0], [0, 5], [0, 0]]},
    )
    assert len(result["tokens"]) == 1
    token = result["tokens"][0]
    assert (token["token"], token["label"], token["start"], token["end"]) == (
        "Paris",
        "LOCATION",
        0,
        5,
    )
    assert 0.9 < token["score"] < 1
