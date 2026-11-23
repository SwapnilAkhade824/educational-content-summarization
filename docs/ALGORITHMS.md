# Algorithms

## TF-IDF
Term Frequency-Inverse Document Frequency vectorization using scikit-learn `TfidfVectorizer`. Produces weighted term-document matrices and top-weighted terms.

## TextRank
Graph-based extractive summarization:
1. Sentence segmentation.
2. TF-IDF sentence vectors.
3. Cosine similarity between sentences.
4. Similarity graph construction.
5. PageRank (NetworkX) for sentence importance.
6. Select top sentences, restore original order.

## VADER
Lexicon-based sentiment analysis for student feedback. Returns compound/positive/negative/neutral scores and labels using standard thresholds (compound >= 0.05 positive, <= -0.05 negative, otherwise neutral).

## NER
spaCy named entity recognition identifies entities (ORG, PERSON, GPE, DATE, etc.) with text, label, and character offsets.

## Definition Extraction
Rule-based regex patterns for structures: "X is defined as...", "X refers to...", "X means...", "X is a/an/the...". Pattern-based, not guaranteed understanding.

## LDA
Latent Dirichlet Allocation using scikit-learn. CountVectorizer (Bag-of-Words) input, configurable topics, iterations, random seed. Produces topic keywords/weights and document-topic distributions.

## LLM Summarization (Google Gemini)
Official Google GenAI SDK (`google-genai`) with model `gemini-3.8-flash`. All configuration loaded from environment variables (`GEMINI_API_KEY`, `GEMINI_MODEL_ID`, `GEMINI_THINKING_LEVEL`, `GEMINI_MAX_OUTPUT_TOKENS`, `GEMINI_TEMPERATURE`, `GEMINI_TIMEOUT_SECONDS`). Supports 5 structured academic summary types: short summary, detailed summary, bullet-point notes, important definitions, and exam revision notes. Long documents are chunked with sentence boundaries and overlap, summarized per chunk, then synthesized. Gracefully bypassed when unconfigured without impacting traditional NLP stages.
