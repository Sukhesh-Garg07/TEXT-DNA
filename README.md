TextDNA — Full-Spectrum NLP Intelligence Engine

"An MRI for text — from basic linguistics to cognitive analysis in one unified pipeline."

What is TextDNA?

TextDNA takes any piece of text and runs it through 5 progressive NLP layers, producing a rich intelligence report that goes far beyond sentiment analysis or named-entity recognition.

Layer	Domain	What it does
1	Linguistic Foundation	Tokenisation, POS tagging, NER, dependency tree depth, readability suite (Flesch, Gunning Fog, SMOG, ARI), Type-Token Ratio
2	Semantic Intelligence	Sentence embeddings, intra-text coherence scoring, semantic drift detection, key concept extraction, central sentence identification
3	Argument Mining & Discourse	Sentence-level argument role classification (claim / evidence / conclusion / counterargument), logical connective analysis, rhetorical device detection (anaphora, tricolon, hyperbole, rhetorical questions), logical fallacy flagging (ad hominem, straw man, false dichotomy, slippery slope …)
4	Cognitive & Psychological	Cognitive bias signal detection (confirmation bias, appeal to emotion, bandwagon effect …), Moral Foundation Theory (MFT) analysis, gain vs. loss framing analysis, passive-voice / agency analysis
5	Generative Intelligence	Implicit assumption extraction, information gap detection, signature trait profiling, TextDNA Score (composite 0–100)
Why is this novel?

Most NLP demos pick one capability. TextDNA is the first open-source tool to combine:

·Argument mining

·Logical fallacy detection

·Moral Foundation Theory scoring

·Cognitive bias signal detection

·Semantic drift analysis

…in a single unified pipeline with a composite quality score.

Installation
bash
# 1. Clone / download the project
git clone https://github.com/Sukhesh-Garg07/TEXT-DNA.git
cd textdna

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    

Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy English model
python -m spacy download en_core_web_sm

First run will automatically download facebook/bart-large-mnli (~1.6 GB)
and sentence-transformers/all-MiniLM-L6-v2 (~90 MB).
These are cached by Hugging Face — subsequent runs are instant.

# Run on the built-in demo text
python textdna.py

# Analyse any plain-text file
python textdna.py myarticle.txt

# Paste your own text interactively
python textdna.py --interactive

The report is printed to the terminal with colour-coded sections and also saved as textdna_report.json for downstream use.

Models used
Model	Purpose	Size
spacy/en_core_web_sm	Tokenisation, POS, NER, dependency parse	~12 MB
sentence-transformers/all-MiniLM-L6-v2	Sentence embeddings	~90 MB
facebook/bart-large-mnli	Zero-shot classification (all advanced layers)	~1.6 GB

All models run locally on CPU — no API keys, no data sent anywhere.

# Skills demonstrated

spaCy · sentence-transformers · transformers / HuggingFace · zero-shot classification ·
NER · dependency parsing · readability metrics · TF-IDF · cosine similarity ·
argument mining · rhetorical analysis · cognitive bias detection ·
Moral Foundation Theory · framing analysis · composite scoring

# License

MIT — use it, extend it, publish it.
