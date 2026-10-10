import re
from typing import List, Dict

from rich.console import Console
console = Console()

# ARGUMENT MINING LAYER
# Tools: facebook/bart-large-mnli (zero-shot classification), custom rule engine
# Covers: sentence-level argument roles (claim / evidence / conclusion …), logical connective analysis, rhetorical device detection, logical fallacy classification

def layer3_argument_discourse(sentences: List[str]) -> Dict:
    from transformers import pipeline as hf_pipeline

    console.print("[bold cyan]▶  Layer 3 — Argument Mining & Discourse[/bold cyan]")

    classifier = hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,  # CPU — change to 0 if you have a GPU
    )

    # Argument role classification
    arg_labels = [
        "claim",
        "supporting evidence",
        "background information",
        "conclusion",
        "counterargument",
        "example",
    ]

    argument_roles = []
    for sent in sentences[:20]:   # cap at 20 for speed
        if len(sent.strip()) > 10:
            res = classifier(sent, arg_labels, multi_label=False)
            argument_roles.append({
                "sentence": sent[:100] + ("…" if len(sent) > 100 else ""),
                "role": res["labels"][0],
                "confidence": round(res["scores"][0], 3),
            })

    # Logical connective analysis
    CONNECTIVES = {
        "causal": ["because", "therefore", "thus", "hence", "consequently", "as a result", "due to"],
        "contrastive": ["however", "but", "although", "nevertheless", "on the other hand", "yet", "despite"],
        "additive": ["furthermore", "moreover", "additionally", "also", "in addition"],
        "conditional": ["if", "unless", "provided that", "assuming", "in case"],
        "conclusive" : ["in conclusion", "to summarize", "finally", "in summary", "overall"],
    }
    full_lower      = " ".join(sentences).lower()
    connective_counts = {k: sum(full_lower.count(w) for w in ws) for k, ws in CONNECTIVES.items()}

    # Rhetorical device detection (rule-based)
    def detect_rhetoric(sents: List[str]) -> List[Dict]:
        found = []
        for sent in sents:
            sl = sent.lower()
            # Rhetorical question
            if sent.strip().endswith("?") and any(w in sl for w in
               ["why", "how", "what", "who", "when", "where", "is it not", "isn't it", "don't you"]):
                found.append({"device": "Rhetorical question", "sentence": sent[:80]})
            # Tricolon (rule of three)
            if re.search(r"\b\w+,\s+\w+,\s+and\s+\w+\b", sent):
                found.append({"device": "Tricolon (rule of three)", "sentence": sent[:80]})
            # Hyperbole markers (≥ 2 extreme words in one sentence)
            hyp = ["never", "always", "everyone", "nobody", "impossible", "perfect",
                   "worst", "best", "all time", "ever", "most", "least"]
            if sum(1 for h in hyp if h in sl) >= 2:
                found.append({"device": "Hyperbole", "sentence": sent[:80]})
        # Anaphora (same word starts 3+ consecutive sentences)
        if len(sents) >= 3:
            starts = [s.split()[0].lower() if s.split() else "" for s in sents]
            for i in range(len(starts) - 2):
                if starts[i] and starts[i] == starts[i + 1] == starts[i + 2]:
                    found.append({
                        "device": "Anaphora",
                        "sentence": f'"{starts[i]}" starts 3 consecutive sentences',
                    })
                    break
        return found

    rhetorical_devices = detect_rhetoric(sentences)

    # Logical fallacy detection (zero-shot)
    fallacy_labels = [
        "ad hominem — personal attack on the speaker",
        "straw man — misrepresenting the opponent's argument",
        "false dichotomy — only two options are presented",
        "slippery slope — assuming a chain of events without evidence",
        "appeal to authority — citing authority instead of evidence",
        "hasty generalization — broad conclusion from limited examples",
        "no logical fallacy",
    ]

    fallacies = []
    for sent in sentences[:15]:
        if len(sent.strip()) > 20:
            res = classifier(sent, fallacy_labels, multi_label=False)
            if res["labels"][0] != "no logical fallacy" and res["scores"][0] > 0.35:
                fallacies.append({
                    "sentence": sent[:100] + ("…" if len(sent) > 100 else ""),
                    "fallacy": res["labels"][0],
                    "confidence": round(res["scores"][0], 3),
                })

    return {
        "argument_roles" : argument_roles,
        "connective_analysis": connective_counts,
        "rhetorical_devices": rhetorical_devices,
        "logical_fallacies" : fallacies,
        "argument_structure_summary": {
            "claims": sum(1 for a in argument_roles if a["role"] == "claim"),
            "evidence" : sum(1 for a in argument_roles if "evidence" in a["role"]),
            "conclusions": sum(1 for a in argument_roles if a["role"] == "conclusion"),
        },
        "_classifier": classifier,  # reused in next two layers; removed before JSON export
    }
