# Dataset Documentation

## Source

This project does not depend on a third-party benchmark dataset. It processes educational documents (PDF/TXT) uploaded by the user at runtime.

A sample input document is included for demonstration and grading:

| File | Description |
|---|---|
| `data/raw/educational_content_nlp_test_document.pdf` | Author-created 7-page educational PDF covering Artificial Intelligence, Machine Learning, Supervised & Unsupervised Learning, Neural Networks, NLP, TextRank & LDA, and NLP Evaluation. Contains definitional sentences ("X is defined as ...") so that rule-based definition extraction, NER, TextRank and LDA can all be demonstrated. |
| `data/evaluation/demo/*.json` | Small author-created evaluation sets (summarization references, concept annotations, sentiment-labelled feedback). Demonstration only. |
| `data/evaluation/templates/*.json` | Empty templates for adding human-labelled evaluation data. |
| `data/evaluation/reports/evaluation_report.json` | Generated evaluation report (ROUGE, sentiment accuracy, concept F1, LDA reproducibility). |

## Formats

- Input documents: PDF (PyMuPDF, page-by-page) and TXT (UTF-8 / UTF-8-SIG with cp1252 and latin-1 fallback). Maximum upload size: 10 MB.
- Evaluation data and reports: JSON.

## Preprocessing Details

1. **Extraction** – page text extracted; empty/scanned pages flagged with warnings; original text preserved.
2. **Sentence tokenization** – NLTK Punkt (`sent_tokenize`); PDF heading lines and line wraps are separated/collapsed.
3. **Word tokenization** – NLTK `word_tokenize`.
4. **Cleaning** – Unicode NFC normalization, control-character removal, lowercasing, whitespace collapse.
5. **Stopword removal** – NLTK English stopwords; negations `no`, `not`, `never` are preserved.
6. **Lemmatization** – WordNet lemmatizer (POS-aware when the tagger is available); technical gerunds such as "learning" are protected.
7. **Vectorization** – TF-IDF on the lemmatized, stopword-filtered tokens; CountVectorizer (bag-of-words) for LDA.

## Licensing

No copyrighted textbook content is included. Users are responsible for having the rights to any documents they upload.
