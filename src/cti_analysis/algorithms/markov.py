"""Markov Entity Smoother — HMM-based entity type smoothing using Viterbi.

Smooths entity type assignments across a document by modeling type transitions
as a Hidden Markov Model. High-confidence type observations influence
neighboring entities to maintain consistency.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from cti_analysis.ontology import TYPES


@dataclass
class SmoothableEntity:
    """Entity with confidence, used for Markov smoothing."""
    text: str
    type: str
    confidence: float = 0.8

    def __hash__(self):
        return hash((self.text.lower(), self.type))

    def __eq__(self, other):
        if not isinstance(other, SmoothableEntity):
            return False
        return self.text.lower() == other.text.lower() and self.type == other.type


class MarkovEntitySmoother:
    """HMM-based entity type smoother using Viterbi decoding.

    Models entity type sequences as a Markov chain where the same type
    is likely to persist (high diagonal in transition matrix) and type
    changes are penalized by alpha.

    Args:
        states: List of entity type strings to model.
        alpha: Transition probability smoothing (0.05-0.5).
               Lower = stricter (types less likely to change).
    """

    def __init__(self, states: Optional[List[str]] = None, alpha: float = 0.1):
        self.states = states or sorted(TYPES) + ["other"]
        n = len(self.states)
        self.alpha = alpha

        # Transition matrix: P(state_t | state_{t-1})
        self.P = np.full((n, n), alpha)
        np.fill_diagonal(self.P, 1.0)
        self.P /= self.P.sum(axis=1, keepdims=True)

        # Initial state distribution (uniform)
        self.pi = np.full(n, 1.0 / n)

    def _state_index(self, entity_type: str) -> int:
        if entity_type in self.states:
            return self.states.index(entity_type)
        return self.states.index("other")

    def viterbi(self, observed_types: List[str],
                confidences: Optional[List[float]] = None) -> List[str]:
        """Run Viterbi to find the optimal type sequence.

        Args:
            observed_types: Observed entity types per position.
            confidences: Optional confidence scores per position.

        Returns:
            List of smoothed entity types.
        """
        n = len(self.states)
        T = len(observed_types)
        if T == 0:
            return []

        confidences = confidences or [0.8] * T

        # Emission probabilities: P(observation | state)
        E = np.zeros((T, n))
        for t in range(T):
            obs_idx = self._state_index(observed_types[t])
            for s in range(n):
                if s == obs_idx:
                    E[t, s] = confidences[t]
                else:
                    E[t, s] = (1.0 - confidences[t]) / (n - 1)

        # Viterbi forward pass (log space for numerical stability)
        V = np.log(self.pi + 1e-9) + np.log(E[0] + 1e-9)
        backpointers = np.zeros((T, n), dtype=int)

        for t in range(1, T):
            V_new = np.empty(n)
            for s in range(n):
                scores = V + np.log(self.P[:, s] + 1e-9) + np.log(E[t, s] + 1e-9)
                backpointers[t, s] = int(np.argmax(scores))
                V_new[s] = np.max(scores)
            V = V_new

        # Backtrack
        path = [int(np.argmax(V))]
        for t in range(T - 1, 0, -1):
            path.append(backpointers[t, path[-1]])
        path = path[::-1]

        return [self.states[p] for p in path]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on common delimiters."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def apply_markov_smoothing(entities: List[SmoothableEntity], text: str,
                           alpha: Optional[float] = None) -> List[SmoothableEntity]:
    """Apply Markov smoothing to entity type assignments.

    Groups entities by sentence, runs Viterbi on the per-sentence type
    sequence, and propagates smoothed types back to entities.
    """
    if not entities:
        return entities

    alpha = alpha or 0.1
    sentences = split_sentences(text)
    if not sentences:
        return entities

    # Aggregate entity types per sentence
    sent_types = []
    sent_confidences = []

    for sent in sentences:
        sent_lower = sent.lower()
        ents_in_sent = [e for e in entities if e.text.lower() in sent_lower]

        if ents_in_sent:
            type_counts: dict[str, int] = defaultdict(int)
            total_conf = 0.0
            for e in ents_in_sent:
                type_counts[e.type] += 1
                total_conf += e.confidence

            top_type = max(type_counts, key=lambda t: type_counts[t])
            avg_conf = total_conf / len(ents_in_sent)
            sent_types.append(top_type)
            sent_confidences.append(avg_conf)
        else:
            sent_types.append("other")
            sent_confidences.append(0.5)

    smoother = MarkovEntitySmoother(alpha=alpha)
    smoothed_types = smoother.viterbi(sent_types, sent_confidences)

    # Apply smoothed types back
    smoothed_entities = []
    for e in entities:
        new_type = e.type
        for i, sent in enumerate(sentences):
            if e.text.lower() in sent.lower():
                new_type = smoothed_types[i]
                break
        smoothed_entities.append(SmoothableEntity(e.text, new_type, e.confidence))

    return smoothed_entities
