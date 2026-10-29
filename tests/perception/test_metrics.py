import pytest

from perception.metrics import MetricInputError, classification_report


def test_report_preserves_literal_confusion_and_lookalike_errors() -> None:
    report = classification_report(
        y_true=[0, 0, 1, 1, 2],
        y_pred=[0, 1, 0, 1, 1],
        labels=("A", "B", "C"),
        lookalike_groups=(("A", "B"),),
        scenarios=[
            {"set": "B"},
            {"set": "B"},
            {"set": "B"},
            {"set": "H"},
            {"set": "H"},
        ],
        scenario_keys=("set",),
    )

    assert report["samples"] == 5
    assert report["accuracy"] == 0.4
    assert report["macro_f1"] == pytest.approx(0.3)
    assert report["confusion_matrix"] == [
        [1, 1, 0],
        [1, 1, 0],
        [0, 1, 0],
    ]
    assert report["per_class"]["A"] == {
        "support": 2,
        "predicted": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert report["per_class"]["C"] == {
        "support": 1,
        "predicted": 0,
        "precision": None,
        "recall": 0.0,
        "f1": 0.0,
    }
    assert report["lookalike_groups"]["A|B"] == {
        "labels": ["A", "B"],
        "samples": 4,
        "correct": 2,
        "confused_within_group": 2,
        "predicted_outside_group": 0,
        "accuracy": 0.5,
    }
    assert report["scenarios"]["set"] == {
        "B": {"samples": 3, "correct": 1, "accuracy": pytest.approx(1 / 3)},
        "H": {"samples": 2, "correct": 1, "accuracy": 0.5},
    }


def test_report_keeps_all_30_classes_even_when_some_have_no_samples() -> None:
    labels = tuple(f"class_{index:02d}" for index in range(30))

    report = classification_report([0], [0], labels, lookalike_groups=())

    assert len(report["confusion_matrix"]) == 30
    assert all(len(row) == 30 for row in report["confusion_matrix"])
    assert report["per_class"]["class_29"] == {
        "support": 0,
        "predicted": 0,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    assert report["macro_f1"] == pytest.approx(1 / 30)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        ([0], [], "same length"),
        ([3], [0], "outside"),
        ([], [], "at least one"),
    ],
)
def test_invalid_metric_inputs_are_rejected(
    y_true: list[int],
    y_pred: list[int],
    message: str,
) -> None:
    with pytest.raises(MetricInputError, match=message):
        classification_report(y_true, y_pred, ("A", "B", "C"), lookalike_groups=())
