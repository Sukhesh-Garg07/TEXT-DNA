from typing import List, Dict

from rich.console import Console
console = Console()

# COGNITIVE & PSYCHOLOGICAL ANALYSIS LAYER
# Tools: facebook/bart-large-mnli (reused), custom lexicon rules
# Covers: cognitive bias signal detection, Moral Foundation Theory (MFT), gain/loss framing analysis, active vs. passive voice (agency)

def layer4_cognitive_analysis(text: str, sentences: List[str], classifier) -> Dict:
    console.print("[bold cyan]▶  Layer 4 — Cognitive & Psychological Analysis[/bold cyan]")

    # Cognitive bias detection (zero-shot on text chunks)
    bias_labels = [
        "confirmation bias — cherry-picking evidence that supports a preset conclusion",
        "anchoring bias — over-relying on the first piece of information",
        "appeal to emotion — manipulating feelings instead of using logic",
        "bandwagon effect — everyone is doing it, so it must be right",
        "false authority — using a credential unrelated to the claim",
        "neutral, unbiased writing",
    ]

    words = text.split()
    chunk_size = 100
    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, min(len(words), 500), chunk_size)
    ]

    bias_signals = []
    for chunk in chunks[:5]:
        res = classifier(chunk, bias_labels, multi_label=True)
        detected = [
            (label, round(score, 3))
            for label, score in zip(res["labels"], res["scores"])
            if score > 0.30 and "neutral" not in label.lower()
        ]
        if detected:
            bias_signals.append({
                "text_segment" : chunk[:80] + "…",
                "detected_biases": detected[:3],
            })

    # Moral Foundation Theory (MFT) lexicon 
    # Jonathan Haidt's 5 moral foundations, with seed lexicons
    MFT = {
        "Care / Harm" : ["harm", "care", "protect", "hurt", "safe", "welfare", "nurture", "cruel", "compassion"],
        "Fairness / Cheating" : ["fair", "unfair", "equal", "justice", "rights", "cheat", "deserve", "bias", "equity"],
        "Loyalty / Betrayal" : ["loyal", "betray", "team", "together", "solidarity", "traitor", "trust", "unity"],
        "Authority / Subversion": ["authority", "obey", "rebel", "tradition", "respect", "order", "hierarchy", "leader"],
        "Purity / Degradation" : ["pure", "sacred", "disgusting", "corrupt", "clean", "moral", "sin", "holy", "degrade"],
    }

    text_lower = text.lower()
    mft_scores = {k: sum(text_lower.count(kw) for kw in kws) for k, kws in MFT.items()}
    total_mft = sum(mft_scores.values()) or 1
    mft_pct = {k: round(v / total_mft * 100, 1) for k, v in mft_scores.items()}
    dominant_mf = max(mft_scores, key=mft_scores.get)

    # Gain / Loss framing
    GAIN = ["gain", "benefit", "improve", "achieve", "success", "win", "opportunity", "advantage", "growth", "better", "positive", "increase", "enhance", "profit"]
    LOSS = ["lose", "risk", "danger", "threat", "fail", "worse", "negative", "decrease","damage", "harm", "cost", "problem", "crisis", "fear", "collapse"]

    gain_n = sum(text_lower.count(w) for w in GAIN)
    loss_n = sum(text_lower.count(w) for w in LOSS)

    if gain_n > loss_n * 1.5:
        framing = "Gain-framed  — emphasises benefits and opportunities"
    elif loss_n > gain_n * 1.5:
        framing = "Loss-framed  — emphasises risks and threats"
    else:
        framing = "Balanced framing"

    # Passive voice / agency check (heuristic)
    passive_markers = [" was ", " were ", " is being ", " are being ", " has been ", " have been ", " was being "]
    passive_count = sum(text_lower.count(m) for m in passive_markers)
    agency_note = (
        "High passive voice — responsibility may be diffused"
        if passive_count > 8
        else "Active voice dominant — clear agents and actions"
    )

    return {
        "cognitive_bias_signals": bias_signals,
        "moral_foundation_analysis": {
            "scores" : mft_scores,
            "percentages": mft_pct,
            "dominant_frame": dominant_mf,
        },
        "framing_analysis": {
            "type": framing,
            "gain_signals" : gain_n,
            "loss_signals" : loss_n,
        },
        "agency_analysis": {
            "passive_count": passive_count,
            "note" : agency_note,
        },
    }
