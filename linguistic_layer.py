from collections import Counter
from typing import  Dict

#for better terminal output
from rich.console import Console
console = Console()

# LINGUISTIC LAYER
# Tools: spaCy (en_core_web_sm), textstat
# Covers: tokenization, lemmatization, POS tagging, NER, dependency tree depth, readability scores, vocabulary richness

def layer1_linguistic_foundation(text: str) -> Dict:
    import spacy
    import textstat

    console.print("[bold cyan]▶  Layer 1 — Linguistic Foundation[/bold cyan]")

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Basic counts
    tokens   = [t.text for t in doc if not t.is_space]
    words    = [t.text for t in doc if t.is_alpha]
    sents    = list(doc.sents)

    # POS distribution
    pos_counts = Counter(t.pos_ for t in doc if t.is_alpha)

    # Named entities
    entities      = [(e.text, e.label_) for e in doc.ents]
    entity_types  = Counter(e.label_ for e in doc.ents)

    # Readability suite
    readability = {
        "flesch_reading_ease"        : textstat.flesch_reading_ease(text),
        "gunning_fog"                : textstat.gunning_fog(text),
        "smog_index"                 : textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "grade_level"                : textstat.text_standard(text),
    }

    # Vocabulary richness (Type-Token Ratio)
    word_lower   = [w.lower() for w in words]
    unique_words = set(word_lower)
    ttr          = len(unique_words) / len(word_lower) if word_lower else 0

    # Average dependency tree depth
    def tree_depth(token, d=0):
        children = list(token.children)
        return d if not children else max(tree_depth(c, d + 1) for c in children)

    roots     = [t for t in doc if t.dep_ == "ROOT"]
    avg_depth = sum(tree_depth(r) for r in roots) / len(roots) if roots else 0

    # Sentence length stats
    sent_lens = [len([t for t in s if t.is_alpha]) for s in sents]

    results = {
        "total_tokens"             : len(tokens),
        "total_words"              : len(words),
        "unique_words"             : len(unique_words),
        "total_sentences"          : len(sents),
        "avg_sentence_length"      : round(sum(sent_lens) / len(sent_lens), 1) if sent_lens else 0,
        "max_sentence_length"      : max(sent_lens) if sent_lens else 0,
        "type_token_ratio"         : round(ttr, 4),
        "avg_dependency_tree_depth": round(avg_depth, 2),
        "pos_distribution"         : dict(pos_counts),
        "named_entities"           : entities[:20],
        "entity_type_distribution" : dict(entity_types),
        "readability"              : readability,
        "sentences"                : [s.text.strip() for s in sents],
        # _doc kept private; removed before JSON export
        "_doc"                     : doc,
    }
    return results
