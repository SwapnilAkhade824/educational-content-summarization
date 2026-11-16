# Academic Demonstration Script

- **Topic:** Educational Content Summarization System
- **Key Concepts:** Extractive / Abstractive Summarization
- **Team Members:**
  - SWAPNIL SANTOSH AKHADE
  - BANSUDE SUDARSHAN SAKHARAM

This script walks through demonstrating the Educational Content Summarization System during project presentation or evaluation.

## 1. Startup & Environment Check
1. Start application: `streamlit run app.py`
2. Point out the simple academic layout:
   - Sidebar: Document upload, sentence/topic controls, "Run Complete Pipeline" button, "Reset" button, and document info.
   - Main page: Sequential numbered pipeline (Stages 1 through 11) followed by isolated Student Feedback Sentiment Analysis.
   - Notice status indicators: `⬜ Not Started`, `🔄 Running`, `✅ Completed`, `⚠️ Warning`, `❌ Failed`.

## 2. Document Processing & Pipeline Execution
1. In the sidebar, upload a sample educational document (`.pdf` or `.txt`).
2. Click **▶ Run Complete Pipeline** (or click the **▶ Run** button at the top right of any stage card; missing prerequisite stages run automatically).
3. Walk through each stage in numbered order:
   - **Stage 1 (Text Extraction)**: Inspect page count, word count, and extracted text.
   - **Stage 2 (Sentence Tokenization)**: Show sentence count and individual segmented sentences.
   - **Stage 3 (Word Tokenization)**: Show total token count and tokenized word list.
   - **Stage 4 (Text Cleaning)**: Show original text vs. cleaned text side-by-side (whitespace collapsed, punctuation/case normalized).
   - **Stage 5 (Stopword Removal)**: Inspect list of removed stopwords vs. remaining tokens (note negations like `not`, `no`, `never` are preserved).
   - **Stage 6 (Lemmatization)**: Inspect Original Token vs. Lemma table showing base word normalization.
   - **Stage 7 (TF-IDF Vectorization)**: Inspect vocabulary size, matrix dimensions, numeric feature matrix, and top weighted terms.
   - **Stage 8 (TextRank Summarization)**:
     - Open Sentence Similarity Matrix expander.
     - Review Graph Construction description.
     - Inspect PageRank scores table.
     - Inspect Ranked and Selected sentences.
     - View final summary and compression ratio.
   - **Stage 9 (Information Extraction)**: Inspect extracted named entities (NER), noun-chunk keyphrases, and rule-based definitions.
   - **Stage 10 (LDA Topic Modelling)**: Inspect topic keywords, weights, document-topic distributions, and dominant topics.
   - **Stage 11 (Gemini Summarization)**:
     - Inspect generated short summary, detailed summary, bullet-point notes, important definitions, and exam revision notes (if API key configured).
     - Emphasize that if Gemini is unconfigured, all stages 1-10 run independently.

## 3. Student Feedback Sentiment Analysis (Separate Section)
1. Scroll down to the Student Feedback Sentiment section (explain why it is isolated from technical content).
2. Enter sample positive feedback ("The explanations and diagrams made this concept very clear!") and click **Analyze Single Feedback**. Show compound score and label.
3. Enter batch feedback and click **Analyze Batch Feedback**. Show distribution (positive/negative/neutral percentages) and detailed table.

## 4. Export Study Notes
1. Under Export Study Notes, preview generated Markdown.
2. Click **Download Study Notes (Markdown)** to save `study_notes.md`.

## 5. Automated Verification
1. Run test suite: `python -m pytest -q` (202 tests passing).
