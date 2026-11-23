# Architecture

## Overview

The application follows a modular src-layout. The Streamlit UI (`app.py`) provides a single-page sequential pipeline demonstrating each NLP stage in numbered order with real, inspectable algorithm outputs. Student feedback sentiment analysis is isolated into a separate section. No NLP algorithm is duplicated in the UI; all core processing delegates to reusable modules under `src/`.

## Sequential Pipeline Stages (Main Page)

1. **Text Extraction**: PyMuPDF (`fitz`) for PDF and multi-encoding decoding for TXT. Outputs raw text, page count, word count, and format warnings.
2. **Sentence Tokenization**: NLTK `sent_tokenize` (using `punkt`). Outputs sentence count and numbered sentences.
3. **Word Tokenization**: NLTK `word_tokenize`. Outputs token count and actual tokens.
4. **Text Cleaning**: Unicode NFC normalization, control character removal, whitespace collapse, case folding. Shows original vs. cleaned text side-by-side.
5. **Stopword Removal**: NLTK English stopwords with protected negations (`no`, `not`, `never`) and non-alphanumeric token filtering. Shows removed stopwords vs. remaining tokens.
6. **Lemmatization**: NLTK `WordNetLemmatizer` with POS-tag mapping (`averaged_perceptron_tagger`). Displays original token vs. lemma table.
7. **TF-IDF Vectorization**: scikit-learn `TfidfVectorizer`. Exposes vocabulary, matrix dimensions, numeric feature matrix, and top weighted terms.
8. **TextRank Summarization**: Cosine similarity graph over TF-IDF vectors + NetworkX PageRank. Displays sentence similarity matrix, graph explanation, PageRank scores, ranked/selected sentences, summary text, and compression ratio.
9. **Information Extraction**: spaCy `en_core_web_sm` NER, frequency-based noun-chunk keyphrases, and regex pattern-based definition extraction.
10. **LDA Topic Modelling**: scikit-learn `LatentDirichletAllocation` over `CountVectorizer` (Bag-of-Words). Shows topic keywords, weights, document-topic distributions, and dominant topics.
11. **Gemini Summarization**: Google AI Studio official `google-genai` SDK using `gemini-3.8-flash`. Produces short summary, detailed summary, bullet-point notes, important definitions, and exam revision notes.

## Student Feedback Sentiment Analysis (Separate Section)

- VADER lexicon-based sentiment analysis (`nltk.sentiment.vader`).
- Isolated from technical document pipeline.
- Supports single entry and batch entry (counts, percentages, per-item scores).

## Modules

- `src/extraction`: PDF/TXT extraction, validation, common interface.
- `src/preprocessing`: cleaning, tokenization, stopwords, lemmatization, TF-IDF.
- `src/summarization`: TextRank, Google Gemini client (`google-genai`), versioned prompts, chunking.
- `src/information_extraction`: entities, keyphrases, definitions.
- `src/sentiment`: VADER feedback sentiment.
- `src/topic_modeling`: LDA.
- `src/utils`: config, logging.
- `src/evaluation`: metrics and runner.
- `src/ui`: Streamlit helpers.
- `tests`: pytest suite.
- `data`: templates, demo data, reports.

## Configuration

- Environment variables only, read centrally in `src/utils/config.py` (`GEMINI_API_KEY`, `GEMINI_MODEL_ID`, `GEMINI_THINKING_LEVEL`, `GEMINI_MAX_OUTPUT_TOKENS`, `GEMINI_TEMPERATURE`, `GEMINI_TIMEOUT_SECONDS`). There is no in-app LLM configuration UI.
- Gemini credentials never stored in the repo; `.env` is git-ignored.

## Error Handling

Modules raise typed exceptions; UI displays messages. Missing optional resources (spaCy model, NLTK data, LLM endpoint) degrade gracefully.
