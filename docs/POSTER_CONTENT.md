# Poster Content

## Title
Educational Content Summarization System

## Key Concepts
Extractive / Abstractive Summarization

## Team Members
- SWAPNIL SANTOSH AKHADE
- BANSUDE SUDARSHAN SAKHARAM

## Problem Statement
Students and educators face lengthy educational documents. Manual summarization, concept extraction and revision-note preparation are time-consuming, and pure LLM tools hide how results are produced.

## Objectives
Extract text from PDF/TXT, preprocess and vectorize it, produce a transparent extractive summary (TextRank), extract entities/keyphrases/definitions, discover topics (LDA), analyze student feedback sentiment (VADER), and generate structured study notes with Google Gemini — exposing every intermediate result in a Streamlit UI.

## Methodology
PDF/TXT extraction (PyMuPDF) -> sentence & word tokenization -> cleaning -> stopword removal (negations kept) -> lemmatization -> TF-IDF -> TextRank (cosine similarity graph + PageRank) -> spaCy NER / noun-chunk keyphrases / rule-based definitions -> LDA topics (CountVectorizer) + Gemini topic interpretation -> Gemini study notes (5 prompt templates). VADER sentiment runs separately on student feedback.

## Architecture
Modular `src/` layout (extraction, preprocessing, summarization, information_extraction, sentiment, topic_modeling, evaluation, ui, utils). The Streamlit UI (`app.py`) only calls these modules and renders each of the 11 stages with live progress and per-stage Run buttons.

## Key Algorithms
TF-IDF, TextRank (PageRank), spaCy NER, noun-chunk keyphrase ranking, regex definition patterns, LDA, VADER, Google Gemini (`google-genai` SDK) with versioned prompt engineering.

## Results
11-stage pipeline verified on a 7-page educational PDF; 17 definitions and ~49 entities extracted; TextRank summary with compression ratio; LDA topics with document-topic distributions; Gemini short/detailed/bullet/definition/revision notes. 202 automated tests pass (GitHub Actions CI).

## Technologies
Python 3.12, Streamlit, PyMuPDF, NLTK, spaCy, scikit-learn, networkx, google-genai, pytest, GitHub Actions, Streamlit Community Cloud.

## Conclusion
A transparent, academically demonstrable NLP pipeline where classical algorithms do the analysis and Generative AI adds structured study notes.
