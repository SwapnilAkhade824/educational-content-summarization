# Deployment Guide

## 1. Local Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # also installs the spaCy en_core_web_sm model
Copy-Item .env.example .env              # then set GEMINI_API_KEY in .env
streamlit run app.py
```

NLTK resources (punkt, stopwords, wordnet, vader_lexicon, POS tagger) are downloaded automatically on the first app run if missing.

## 2. Push to GitHub

```powershell
git add .
git commit -m "Educational Content Summarization System"
gh repo create educational-content-summarization --public --source . --remote origin --push
```

Before pushing, confirm the API key is not tracked:

```powershell
git check-ignore -v .env        # must print a .gitignore rule
git ls-files | Select-String ".env$"   # must print nothing
```

## 3. Deploy on Streamlit Community Cloud (free)

1. Open https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Repository: `SwapnilAkhade824/educational-content-summarization`, Branch: `main`, Main file path: `app.py`.
4. Open **Advanced settings**:
   - Python version: **3.12**
   - Secrets (TOML) – paste:

     ```toml
     GEMINI_API_KEY = "your-google-ai-studio-key"
     GEMINI_MODEL_ID = "gemini-3.8-flash"
     GEMINI_THINKING_LEVEL = "low"
     GEMINI_MAX_OUTPUT_TOKENS = "8192"
     GEMINI_TEMPERATURE = "0.2"
     GEMINI_TIMEOUT_SECONDS = "120"
     ```

     Streamlit Cloud exposes root-level secrets as environment variables, which `src/utils/config.py` reads. No code change is needed.
5. Click **Deploy**. The first build takes a few minutes (spaCy + model install). The live URL looks like `https://<app-name>.streamlit.app`.

Every `git push` to `main` redeploys the app automatically.

## 4. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | _(empty)_ | Google AI Studio key. Without it, Stages 1–10 and VADER still work; Stage 11 is disabled. |
| `GEMINI_MODEL_ID` | `gemini-3.8-flash` | Gemini model identifier. |
| `GEMINI_THINKING_LEVEL` | `low` | Thinking level. |
| `GEMINI_MAX_OUTPUT_TOKENS` | `8192` | Max generated tokens. |
| `GEMINI_TEMPERATURE` | `0.2` | Sampling temperature. |
| `GEMINI_TIMEOUT_SECONDS` | `120` | Request timeout. |
| `GEMINI_MAX_SINGLE_REQUEST_TOKENS` | `16000` | Documents larger than this are chunked. |

## 5. Continuous Integration

`.github/workflows/ci.yml` installs dependencies and runs the full pytest suite on every push and pull request.
