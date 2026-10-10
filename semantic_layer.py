from typing import List, Dict

#for better terminal output
from rich.console import Console
console = Console()

# SEMANTIC INTELLIGENCE LAYER
# Tools: sentence-transformers (all-MiniLM-L6-v2), scikit-learn TF-IDF
# Covers: sentence embeddings, intra-text semantic coherence, semantic drift detection, key concept extraction, most "central" sentence identification

def layer2_semantic_intelligence(sentences: List[str]) -> Dict:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    console.print("[bold cyan]▶  Layer 2 — Semantic Intelligence[/bold cyan]")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    # Consecutive-sentence cosine similarity → coherence
    coherence_scores = [
        float(cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0])
        for i in range(len(embeddings) - 1)
    ]
    avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1.0

    # Drift points = sentences where coherence drops > 1 std-dev below mean
    if coherence_scores:
        mean_c = avg_coherence
        std_c  = float(np.std(coherence_scores))
        drift_points = [
            {
                "position" : i + 1,
                "sentence" : sentences[i + 1] if i + 1 < len(sentences) else "",
                "coherence_drop" : round(mean_c - score, 4),
            }
            for i, score in enumerate(coherence_scores)
            if score < mean_c - std_c
        ]
    else:
        drift_points = []

    # Key concepts via TF-IDF across sentences
    key_concepts = []
    if len(sentences) > 1:
        try:
            tfidf = TfidfVectorizer(max_features=15, stop_words="english", ngram_range=(1, 2))
            tfidf.fit_transform(sentences)
            key_concepts = tfidf.get_feature_names_out().tolist()
        except Exception:
            pass

    # Global semantic diversity (1 − mean pairwise similarity)
    semantic_diversity = 0.0
    if len(embeddings) > 1:
        all_sims = cosine_similarity(embeddings)
        np.fill_diagonal(all_sims, 0)
        semantic_diversity = float(1 - np.mean(all_sims))

    # Most "central" sentence (highest average similarity to all others)
    central_idx = int(np.argmax(cosine_similarity(embeddings).mean(axis=1))) if len(embeddings) > 1 else 0
    central_sentence = sentences[central_idx] if sentences else ""

    return {
        "avg_semantic_coherence" : round(avg_coherence, 4),
        "semantic_diversity_score" : round(semantic_diversity, 4),
        "coherence_per_sentence" : [round(s, 4) for s in coherence_scores],
        "semantic_drift_points" : drift_points,
        "key_concepts" : key_concepts,
        "central_sentence" : central_sentence,
        "central_sentence_idx" : central_idx,
        "_embeddings" : embeddings,   # removed before JSON export
    }


