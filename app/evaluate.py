"""Measure the local model honestly.

Run:  python -m app.evaluate

- Category: 5-fold cross-validation, so every ticket is predicted by a model
  that never saw it during training.
- Priority: the rules were written from tickets T001-T070, so they are scored
  separately on T071-T105, which they were not written from.
"""

from __future__ import annotations

from collections import Counter

from .classifier import CATEGORIES, NaiveBayesClassifier, load_tickets, score_priority

RULES_WRITTEN_FROM = 70


def cross_validate_category(folds: int = 5) -> tuple[float, Counter]:
    tickets = load_tickets()
    correct, confusion = 0, Counter()
    for k in range(folds):
        train = [t for i, t in enumerate(tickets) if i % folds != k]
        test = [t for i, t in enumerate(tickets) if i % folds == k]
        model = NaiveBayesClassifier().fit([t.text for t in train], [t.category for t in train])
        for t in test:
            pred, _ = model.predict(t.text)
            correct += pred == t.category
            confusion[(t.category, pred)] += 1
    return correct / len(tickets), confusion


def priority_accuracy() -> tuple[float, float]:
    tickets = load_tickets()
    seen, unseen = tickets[:RULES_WRITTEN_FROM], tickets[RULES_WRITTEN_FROM:]

    def acc(ts):
        return sum(score_priority(t.text).priority == t.priority for t in ts) / len(ts)

    return acc(seen), acc(unseen)


def main() -> None:
    cat_acc, confusion = cross_validate_category()
    seen, unseen = priority_accuracy()
    print(f"Category accuracy (5-fold cross-validation): {cat_acc:.0%}")
    print(f"Priority rules on tickets they were written from: {seen:.0%}")
    print(f"Priority rules on new tickets: {unseen:.0%}\n")
    print("Category per class (correct / total):")
    for c in CATEGORIES:
        total = sum(v for (a, _), v in confusion.items() if a == c)
        print(f"  {c:<24} {confusion[(c, c)]:>2} / {total}")


if __name__ == "__main__":
    main()
