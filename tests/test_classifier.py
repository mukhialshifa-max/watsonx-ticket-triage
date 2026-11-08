import pytest

from app.classifier import CATEGORIES, NaiveBayesClassifier, load_tickets, score_priority, tokenize
from app.evaluate import cross_validate_category, priority_accuracy


def test_tokenize():
    assert tokenize("Labels are not printing on the printers") == ["label", "not", "print", "printer"]


def test_training_data_is_valid_and_balanced():
    tickets = load_tickets()
    assert len(tickets) == 105
    assert {t.category for t in tickets} == set(CATEGORIES)
    assert len({t.id for t in tickets}) == len(tickets)
    counts = {c: sum(t.category == c for t in tickets) for c in CATEGORIES}
    assert set(counts.values()) == {15}


def test_naive_bayes_probabilities_sum_to_one():
    model = NaiveBayesClassifier().fit(["printer jam", "password locked"], ["Devices", "Access"])
    probs = model.predict_proba("my printer")
    assert pytest.approx(sum(probs.values())) == 1.0
    assert model.predict("printer is jammed")[0] == "Devices"


def test_naive_bayes_rejects_bad_input():
    with pytest.raises(ValueError):
        NaiveBayesClassifier().fit([], [])


def test_cross_validated_accuracy_beats_chance_by_a_wide_margin():
    accuracy, _ = cross_validate_category()
    assert accuracy >= 0.65  # chance with 7 balanced classes is about 14%


def test_priority_rules_generalize_to_new_tickets():
    _, unseen = priority_accuracy()
    assert unseen >= 0.65


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Medication record shows the wrong dose", "P1"),
        ("The EHR is down for the whole hospital", "P1"),
        ("Downtime viewer will not open", "P1"),
        ("I am locked out of my account", "P2"),
        ("How do I change my default printer", "P4"),
        ("Spell check is not working in notes", "P3"),
    ],
)
def test_priority_rules(text, expected):
    assert score_priority(text).priority == expected
