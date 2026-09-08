"""
Before asking the LLM "do these two facts agree or conflict?", we need a
cheap way to find which pairs of facts are even *about the same thing*
across potentially hundreds of extracted facts. Calling an LLM on every
pair is O(n^2) and expensive; comparing embeddings via a downloaded model
isn't available in this offline sandbox, so we use TF-IDF + cosine
similarity over "subject + attribute + fact_type" as a fast, dependency-light
retrieval step. This is a documented trade-off (see README) - swapping in a
real embedding model is the obvious upgrade.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def _fact_text(fact):
    parts = [fact["subject"], fact["attribute"], fact.get("fact_type") or ""]
    return " ".join(p for p in parts if p)


def find_candidates(new_facts, existing_facts, top_k=3, min_similarity=0.25):
    """
    For each fact in new_facts, find up to top_k facts in existing_facts
    (must be from a *different* document) that are plausibly about the same
    subject/attribute, using TF-IDF cosine similarity.

    Returns: list of (new_fact, existing_fact, similarity) tuples.
    """
    if not existing_facts or not new_facts:
        return []

    corpus = [_fact_text(f) for f in existing_facts]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    existing_matrix = vectorizer.fit_transform(corpus)

    new_texts = [_fact_text(f) for f in new_facts]
    new_matrix = vectorizer.transform(new_texts)

    sims = cosine_similarity(new_matrix, existing_matrix)

    results = []
    for i, new_fact in enumerate(new_facts):
        row = sims[i]
        # candidates from a different document than this new fact
        idxs = np.argsort(-row)
        count = 0
        for idx in idxs:
            if count >= top_k:
                break
            score = row[idx]
            if score < min_similarity:
                break
            existing_fact = existing_facts[idx]
            if existing_fact["document_id"] == new_fact["document_id"]:
                continue
            results.append((new_fact, existing_fact, float(score)))
            count += 1
    return results
