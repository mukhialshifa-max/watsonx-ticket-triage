"""Local triage engine: a Naive Bayes model for category and explainable rules for priority.

Naive Bayes is written in plain Python so the project stays lightweight and every
step is easy to explain. Priority uses rules, not ML, because support teams need
to see exactly why a ticket was marked urgent.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tickets.csv"

CATEGORIES = [
    "Access & Login",
    "Scheduling",
    "Registration",
    "Clinical Documentation",
    "Results & Inbox",
    "Printing & Devices",
    "Network & Connectivity",
]
PRIORITIES = ["P1", "P2", "P3", "P4"]

STOP_WORDS = {
    "a", "an", "and", "the", "to", "of", "in", "on", "for", "is", "it", "i", "my",
    "me", "we", "our", "this", "that", "with", "at", "be", "are", "was", "so", "do",
    "how", "after", "from", "every", "since", "but", "or", "by", "their", "they",
    "have", "has", "will", "all", "when", "even", "though", "were", "there",
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for w in words:
        if w in STOP_WORDS or len(w) < 2:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                w = w[: -len(suffix)]
                break
        out.append(w)
    return out


@dataclass(frozen=True)
class Ticket:
    id: str
    text: str
    category: str
    priority: str


def load_tickets(path: str | Path = DATA_PATH) -> list[Ticket]:
    with open(path, newline="", encoding="utf-8") as f:
        return [Ticket(r["id"], r["text"], r["category"], r["priority"]) for r in csv.DictReader(f)]


class NaiveBayesClassifier:
    """Multinomial Naive Bayes with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_counts: Counter[str] = Counter()
        self.word_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def fit(self, texts: list[str], labels: list[str]) -> "NaiveBayesClassifier":
        if not texts or len(texts) != len(labels):
            raise ValueError("texts and labels must be non-empty and the same length")
        for text, label in zip(texts, labels):
            tokens = tokenize(text)
            self.class_counts[label] += 1
            self.word_counts[label].update(tokens)
            self.vocab.update(tokens)
        return self

    def predict_proba(self, text: str) -> dict[str, float]:
        tokens = [t for t in tokenize(text) if t in self.vocab]
        total_docs = sum(self.class_counts.values())
        v = len(self.vocab)
        log_scores = {}
        for label, n_docs in self.class_counts.items():
            counts = self.word_counts[label]
            total_words = sum(counts.values())
            score = math.log(n_docs / total_docs)
            for t in tokens:
                score += math.log((counts[t] + self.alpha) / (total_words + self.alpha * v))
            log_scores[label] = score
        # Softmax turns log scores into probabilities that add up to 1.
        top = max(log_scores.values())
        exp = {k: math.exp(s - top) for k, s in log_scores.items()}
        z = sum(exp.values())
        return {k: v / z for k, v in sorted(exp.items(), key=lambda kv: kv[1], reverse=True)}

    def predict(self, text: str) -> tuple[str, float]:
        label, prob = next(iter(self.predict_proba(text).items()))
        return label, round(prob, 3)


# --- Priority rules -----------------------------------------------------------------

SAFETY_PHRASES = [
    "wrong patient", "wrong dose", "wrong date of birth", "merged", "critical",
    "allergies", "attached to the wrong",
]
SCOPE_WORDS = [
    "whole", "everyone", "entire", "hospital", "department", "operating room",
    "all clerks", "for all", "all users", "nobody",
]
OUTAGE_WORDS = ["down", "outage", "cannot log in", "cannot book", "not loading"]
REQUEST_STARTS = ("how do i", "how can i", "request", "please add", "need a", "where do i", "i need to", "can you add")
BLOCKED_WORDS = [
    "locked", "cannot log in", "cannot see", "cannot sign", "disappeared", "not appearing",
    "not printing", "not showing", "error when", "freez", "duplicate", "disconnect",
    "stopped", "not reading", "pending", "disabled", "patients waiting", "slow for everyone",
    "extremely slow", "gives an error",
]


@dataclass(frozen=True)
class PriorityResult:
    priority: str
    reason: str


def score_priority(text: str) -> PriorityResult:
    t = " " + text.lower() + " "
    if any(p in t for p in SAFETY_PHRASES):
        return PriorityResult("P1", "Possible patient-safety risk")
    has_scope = any(w in t for w in SCOPE_WORDS)
    if "downtime" in t or (has_scope and any(w in t for w in OUTAGE_WORDS)):
        return PriorityResult("P1", "System outage affecting many users")
    if t.strip().startswith(REQUEST_STARTS):
        return PriorityResult("P4", "How-to question or service request")
    if any(w in t for w in BLOCKED_WORDS):
        return PriorityResult("P2", "User or team is blocked from working")
    return PriorityResult("P3", "Issue with a workaround or limited impact")
