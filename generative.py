from typing import List, Dict
from rich.console import Console


console = Console()

# GENERATIVE INTELLIGENCE LAYER
# Tools: facebook/bart-large-mnli (reused)
# Covers: implicit assumption mining, information gap analysis, signature trait extraction, TextDNA composite score (0–100)

def layer5_generative_intelligence(
    text: str,
    sentences: List[str],
    l1: Dict, l2: Dict, l3: Dict, l4: Dict,
    classifier,
) -> Dict:
    console.print("[bold cyan]▶  Layer 5 — Generative Intelligence[/bold cyan]")

    # Implicit assumption mining
    assumption_types = [
        "assumes the reader already agrees with the premise",
        "assumes causation from correlation",
        "assumes the proposed solution is the only viable one",
        "assumes the reader has significant background knowledge",
        "assumes a single homogeneous audience",
        "makes no strong assumptions",
    ]

    assumptions_found = []
    for chunk in [" ".join(sentences[:5]), " ".join(sentences[-5:])]:
        if chunk.strip():
            res = classifier(chunk, assumption_types, multi_label=True)
            for label, score in zip(res["labels"], res["scores"]):
                if score > 0.35 and "no strong" not in label:
                    assumptions_found.append({"assumption": label, "confidence": round(score, 3)})

    # Deduplicate
    seen, unique_assumptions = set(), []
    for a in assumptions_found:
        if a["assumption"] not in seen:
            seen.add(a["assumption"])
            unique_assumptions.append(a)

    # Information gap detectio
    gap_labels = [
        "Who is responsible or involved?",
        "What are the long-term consequences?",
        "What is the evidence or primary source?",
        "What are the alternatives or counterarguments?",
        "When and where does this apply?",
        "What does this cost or require?",
    ]

    gap_res = classifier(text[:512], gap_labels, multi_label=True)
    info_gaps = [
        {"gap": label, "relevance": round(score, 3)}
        for label, score in zip(gap_res["labels"], gap_res["scores"])
        if score > 0.40
    ]

    # TextDNA Score (composite, 0–100)
    # Each sub-dimension contributes up to 25 points.

    # 1. Linguistic complexity  (dependency tree depth, normalised to 25)
    linguistic_score = min(l1.get("avg_dependency_tree_depth", 0) * 10, 25)

    # 2. Vocabulary richness (TTR * 25)
    vocab_score = l1.get("type_token_ratio", 0) * 25

    # 3. Semantic coherence (avg cosine similarity * 25)
    coherence_score = l2.get("avg_semantic_coherence", 0) * 25

    # 4. Argument structure  (proportional to presence of claims, evidence, conclusions)
    args = l3.get("argument_structure_summary", {})
    arg_raw = (
        args.get("claims", 0) * 2
        + args.get("evidence", 0) * 3
        + args.get("conclusions", 0) * 2
    )
    argument_score = min(arg_raw, 25)

    text_dna_score = round(linguistic_score + vocab_score + coherence_score + argument_score, 1)

    # Signature traits
    traits = []
    if l1.get("type_token_ratio", 0) > 0.70:
        traits.append("High lexical diversity")
    if l2.get("avg_semantic_coherence", 0) > 0.70:
        traits.append("Strong semantic coherence")
    if l2.get("semantic_diversity_score", 0) > 0.50:
        traits.append("Semantically diverse content")
    if l3.get("rhetorical_devices"):
        traits.append(f"{len(l3['rhetorical_devices'])} rhetorical device(s) detected")
    if l3.get("logical_fallacies"):
        traits.append(f"{len(l3['logical_fallacies'])} logical fallacy signal(s) flagged")
    dom = l4["moral_foundation_analysis"]["dominant_frame"]
    if l4["moral_foundation_analysis"]["scores"].get(dom, 0) > 2:
        traits.append(f"Dominant moral frame: {dom}")
    if "Loss-framed" in l4["framing_analysis"]["type"]:
        traits.append("Loss-framed persuasion strategy")
    elif "Gain-framed" in l4["framing_analysis"]["type"]:
        traits.append("Gain-framed persuasion strategy")

    return {
        "implicit_assumptions": unique_assumptions[:5],
        "information_gaps"    : info_gaps,
        "text_dna_score"      : text_dna_score,
        "text_dna_breakdown"  : {
            "linguistic_complexity": round(linguistic_score, 1),
            "vocabulary_richness"  : round(vocab_score, 1),
            "semantic_coherence"   : round(coherence_score, 1),
            "argument_structure"   : round(argument_score, 1),
        },
        "signature_traits": traits,
    }
