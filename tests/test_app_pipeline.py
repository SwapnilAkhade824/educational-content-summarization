"""Tests for Streamlit application stages and UI execution pipeline."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

import app


def test_app_init_state() -> None:
    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        for key in app._STAGE_KEYS:
            assert mock_state[key] == "Not Started"
        assert mock_state["pipeline_ran"] is False


def test_app_badge() -> None:
    assert "✅" in app._badge("Completed")
    assert "🔄" in app._badge("Running")
    assert "⚠️" in app._badge("Warning")
    assert "❌" in app._badge("Failed")
    assert "⬜" in app._badge("Not Started")


def test_app_run_extraction_valid_txt() -> None:
    content = b"Natural language processing enables computers to understand human language."
    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        success = app._run_extraction(content, "test.txt")
        assert success is True
        assert mock_state["stage_extraction"] == "Completed"
        assert "Natural language processing" in mock_state["raw_text"]
        assert mock_state["extracted_data"].page_count == 1


def test_app_run_extraction_empty_file() -> None:
    content = b""
    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        success = app._run_extraction(content, "empty.txt")
        assert success is False
        assert mock_state["stage_extraction"] == "Warning"


def test_app_run_extraction_invalid_file() -> None:
    content = b"invalid pdf header"
    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        with patch("streamlit.error") as mock_err:
            success = app._run_extraction(content, "invalid.pdf")
            assert success is False
            assert mock_state["stage_extraction"] == "Failed"
            mock_err.assert_called_once()


def test_app_run_sentence_tokenization() -> None:
    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        # Empty text warning
        assert app._run_sentence_tokenization() is False
        assert mock_state["stage_sentence_tok"] == "Warning"

        mock_state["raw_text"] = "First sentence. Second sentence."
        assert app._run_sentence_tokenization() is True
        assert mock_state["stage_sentence_tok"] == "Completed"
        assert len(mock_state["sentences"]) == 2


def test_app_run_word_tokenization() -> None:
    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        assert app._run_word_tokenization() is False
        assert mock_state["stage_word_tok"] == "Warning"

        mock_state["raw_text"] = "Hello world!"
        assert app._run_word_tokenization() is True
        assert mock_state["stage_word_tok"] == "Completed"
        assert "Hello" in mock_state["word_tokens"]


def test_app_run_text_cleaning() -> None:
    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        assert app._run_text_cleaning() is False
        assert mock_state["stage_cleaning"] == "Warning"

        mock_state["raw_text"] = "Hello   World\n!"
        assert app._run_text_cleaning() is True
        assert mock_state["stage_cleaning"] == "Completed"
        assert mock_state["cleaned_text"] == "hello world !"


def test_app_run_stopword_removal() -> None:
    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        assert app._run_stopword_removal() is False
        assert mock_state["stage_stopword"] == "Warning"

        mock_state["word_tokens"] = ["the", "quick", "brown", "fox", "not"]
        assert app._run_stopword_removal() is True
        assert mock_state["stage_stopword"] == "Completed"
        assert "the" in mock_state["removed_stopwords"]
        assert "quick" in mock_state["filtered_tokens"]
        assert "not" in mock_state["filtered_tokens"]  # protected negation


def test_app_run_lemmatization() -> None:
    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        assert app._run_lemmatization() is False
        assert mock_state["stage_lemma"] == "Warning"

        mock_state["filtered_tokens"] = ["running", "studies", "better"]
        assert app._run_lemmatization() is True
        assert mock_state["stage_lemma"] == "Completed"
        assert len(mock_state["lemmatized_tokens"]) == 3
        assert len(mock_state["lemma_pairs"]) == 3


def test_app_run_tfidf() -> None:
    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        assert app._run_tfidf() is False
        assert mock_state["stage_tfidf"] == "Warning"

        mock_state["cleaned_text"] = "machine learning models learn from training data"
        assert app._run_tfidf() is True
        assert mock_state["stage_tfidf"] == "Completed"
        assert mock_state["tfidf_result"].matrix_shape[0] == 1
        assert len(mock_state["tfidf_result"].feature_names) > 0


def test_app_run_textrank_stage() -> None:
    text = (
        "Machine learning is a subset of artificial intelligence. "
        "It enables computers to learn from data without explicit programming. "
        "Data preprocessing is an essential step in building robust models."
    )
    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        assert app._run_textrank_stage() is False
        assert mock_state["stage_textrank"] == "Warning"

        mock_state["raw_text"] = text
        assert app._run_textrank_stage(num_sentences=2) is True
        assert mock_state["stage_textrank"] == "Completed"
        assert len(mock_state["textrank_result"].selected_sentences) <= 2
        assert mock_state["similarity_matrix"] is not None


def test_app_run_info_extraction() -> None:
    text = "Machine learning is defined as a field of study in artificial intelligence. Google and IBM develop AI."
    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        assert app._run_info_extraction() is False
        assert mock_state["stage_info_extraction"] == "Warning"

        mock_state["raw_text"] = text
        assert app._run_info_extraction() is True
        assert mock_state["stage_info_extraction"] in ("Completed", "Warning")
        assert len(mock_state["concepts_result"].definitions) > 0


def test_app_run_lda_stage() -> None:
    text = (
        "Machine learning is a subset of artificial intelligence. "
        "Natural language processing enables computers to understand human language. "
        "Deep learning uses neural networks with many hidden layers."
    )
    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        assert app._run_lda_stage() is False
        assert mock_state["stage_lda"] == "Warning"

        mock_state["raw_text"] = text
        assert app._run_lda_stage(n_topics=2) is True
        assert mock_state["stage_lda"] in ("Completed", "Warning")
        assert mock_state["lda_result"] is not None


def test_app_run_gemini_stage_unconfigured() -> None:
    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        with patch("app.load_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(is_configured=False)
            assert app._run_gemini_stage() is False
            assert mock_state["stage_gemini"] == "Warning"


def test_app_document_change_invalidates_stale_state() -> None:
    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        doc1 = b"Document one content about supervised learning."
        doc2 = b"Document two content about unsupervised clustering."

        changed1 = app._ensure_fresh_document("doc1.txt", doc1)
        assert changed1 is False
        app._run_extraction(doc1, "doc1.txt")
        mock_state["gemini_results"] = {"short": "Old doc1 summary"}
        mock_state["sentiment_single"] = "preserved_sentiment"

        # Switching to doc2 must invalidate document pipeline state while keeping sentiment
        changed2 = app._ensure_fresh_document("doc2.txt", doc2)
        assert changed2 is True
        assert mock_state["stage_extraction"] == "Not Started"
        assert mock_state["raw_text"] is None
        assert mock_state["gemini_results"] is None
        assert mock_state["sentiment_single"] == "preserved_sentiment"


def test_app_stage_step_hooks_and_logs() -> None:
    content = (
        b"Machine learning is defined as a subset of artificial intelligence. "
        b"Neural networks are known as deep learning models. "
        b"Students should not ignore data preprocessing."
    )
    steps_recorded: list[tuple[int, int, str]] = []

    def _hook(step: int, total: int, msg: str) -> None:
        steps_recorded.append((step, total, msg))

    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        assert app._run_extraction(content, "notes.txt", step_hook=_hook) is True
        assert app._run_sentence_tokenization(step_hook=_hook) is True
        assert app._run_word_tokenization(step_hook=_hook) is True
        assert app._run_text_cleaning(step_hook=_hook) is True
        assert app._run_stopword_removal(step_hook=_hook) is True
        assert app._run_lemmatization(step_hook=_hook) is True
        assert app._run_tfidf(step_hook=_hook) is True
        assert app._run_textrank_stage(num_sentences=2, step_hook=_hook) is True
        assert app._run_info_extraction(step_hook=_hook) is True
        assert app._run_lda_stage(n_topics=2, step_hook=_hook) is True

        assert len(steps_recorded) >= 15
        assert len(mock_state["stage_logs"]["stage_textrank"]) == 5
        assert len(mock_state["stage_logs"]["stage_lda"]) == 4
        assert len(mock_state["stage_logs"]["stage_info_extraction"]) >= 3
        # Verify lowercase lemmatization unified 'Models' -> 'model' in TF-IDF vocabulary
        vocab = set(mock_state["tfidf_result"].feature_names)
        assert "model" in vocab
        assert "Models" not in vocab
        assert "not" in vocab


def test_app_gemini_stage_phase_progress_hook() -> None:
    phases_seen: list[tuple[int, int, str]] = []

    def _gemini_hook(phase_idx: int, total_phases: int, msg: str, info: dict) -> None:
        phases_seen.append((phase_idx, total_phases, msg))

    with patch("streamlit.session_state", {}) as mock_state:
        app._init_state()
        mock_state["raw_text"] = "Machine learning enables computers to learn from data."
        with patch("app.load_config") as mock_cfg, patch("app.count_tokens") as mock_tok, patch("app.generate_summary") as mock_gen:
            mock_cfg.return_value = MagicMock(
                is_configured=True,
                model_id="gemini-3.8-flash",
                thinking_level="low",
                max_single_request_tokens=16000,
            )
            mock_tok.return_value = 42

            def _fake_gen(text: str, summary_type: str = "short", **kwargs):
                cb = kwargs.get("progress_callback")
                if cb:
                    cb({"current_chunk": 1, "total_chunks": 1, "completed_chunks": 1, "retry_status": "Completed"})
                return MagicMock(summary_text=f"Generated {summary_type}", chunk_count=1)

            mock_gen.side_effect = _fake_gen
            assert app._run_gemini_stage(step_hook=_gemini_hook) is True
            assert mock_state["stage_gemini"] == "Completed"
            assert mock_state["gemini_token_count"] == 42
            assert len(mock_state["gemini_results"]) == 5
            assert len(phases_seen) == 5
            assert [p[0] for p in phases_seen] == [1, 2, 3, 4, 5]


