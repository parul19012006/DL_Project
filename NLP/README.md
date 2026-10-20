# Hinglish Sentiment Analyzer

A small NLP project that classifies **Hinglish** text (Hindi written in Roman script, mixed with English) as **positive**, **negative** or **neutral**. It also tags every word as Hindi, English or unknown and reports the language mix of the sentence.

The project has two parts:

- **`hinglish_nlp_simple.ipynb`** is the notebook. It covers the dataset, preprocessing, language tagging, model training, evaluation and plots.
- **`app.py` + `index.html`** is a Flask web app. It trains the same models at startup and serves a one-page analyzer.

> Example input: `Yaar ye movie bahut amazing thi, must watch hai!`

---

## Features

- **Hinglish-aware cleaning:** lowercasing, removal of links, @mentions and #hashtags, collapsing of stretched letters (`yaaarrr` becomes `yaar`), and a combined Hindi + English stopword list.
- **Two feature sets stacked together:**
  - word TF-IDF (unigrams and bigrams, up to 1,500 features)
  - character TF-IDF (`char_wb`, 3 to 5 grams, up to 1,000 features), which helps with spelling variants like *bahut / bohot / bhot*
- **Two classifiers:** Logistic Regression and a linear SVM. The web UI lets you switch between them.
- **Word-level language tagging:** a lexicon-based tagger labels each word `HI`, `EN` or `UNK`.
- **Evaluation:** accuracy, precision/recall/F1, a confusion matrix, 5-fold cross-validation and top-weighted features per class.

---

## Project structure

```
NLP/
├── app.py                       # Flask server: trains models on startup, exposes the API
├── index.html                   # Single-page UI (plain HTML + JavaScript, no framework)
├── hinglish_nlp_simple.ipynb    # End-to-end notebook: data, EDA, training, evaluation
└── README.md
```

---

## How it works

```
raw text
   │
   ▼
clean()            lowercase → strip links/@/# → collapse repeated letters
   │               → remove punctuation → drop stopwords and 1-letter tokens
   ▼
TF-IDF (word 1-2 grams)  ┐
                         ├─ stacked side by side (scipy hstack)
TF-IDF (char 3-5 grams)  ┘
   │
   ▼
Logistic Regression  or  Linear SVM  →  label + class probabilities

(in parallel, on the original text)
tag_word() → HI / EN / UNK per word → language-mix percentages
```

### Dataset

The data is a small, hand-written set of **45 Hinglish sentences**, split evenly: 15 positive, 15 negative and 15 neutral. It is split 75/25 with a stratified split (`random_state=42`), which gives **33 training and 12 test sentences**. The data is defined directly inside `app.py` and the notebook, so there is nothing to download.

### Language tagging

Each word is lower-cased, stripped of `. , ! ?`, and then tagged:

1. in the Hindi lexicon, a hand-written set of about 45 common Roman-Hindi words, gives `HI`
2. in NLTK's English word list and longer than 2 characters gives `EN`
3. anything else gives `UNK`

---

## Results

Figures below are from the notebook's saved outputs.

| Model | Hold-out accuracy (12 test sentences) | Macro F1 | 5-fold CV accuracy (mean ± std) |
|---|---|---|---|
| Logistic Regression | 75.0% | 0.76 | 57.8% ± 13.0% |
| SVM (linear) | 83.3% | 0.84 | 51.1% ± 16.6% |

**Read these numbers with care.** The hold-out set is only 12 sentences, so one sentence changes the score by about 8 points. The cross-validation estimate is the more reliable of the two, and it is noticeably lower, with the ranking of the two models reversed. With 45 training examples, the models are a proof of concept, not a production-grade classifier.

---

## Getting started

### Requirements

- Python 3
- An internet connection on the **first run**, because the app downloads the NLTK `stopwords` and `words` corpora

### Install

```bash
cd NLP
pip install flask nltk scikit-learn scipy pandas
```

For the notebook, also install:

```bash
pip install numpy matplotlib seaborn wordcloud
```

### Run the web app

```bash
python app.py
```

Then open **http://127.0.0.1:5000**. The terminal prints the hold-out accuracy of both models once training finishes. Training happens on every startup and takes a moment, and nothing is saved to disk.

### Run the notebook

Open `hinglish_nlp_simple.ipynb` in Jupyter or Google Colab and run all cells. The first code cell installs its own dependencies. It saves three figures next to the notebook: `lang_mix.png`, `results.png` and `top_features.png`.

> **Note:** the notebook as committed contains a saved error from a run where `wordcloud` was not installed (the word-cloud panel in the results figure is missing). Install `wordcloud`, restart the kernel and run all cells to regenerate it.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI (`index.html`) |
| `POST` | `/predict` | Classifies a sentence |
| `GET` | `/model_info` | Returns training metadata and accuracies |

### `POST /predict`

Request body:

```json
{ "text": "Yaar ye movie bahut amazing thi!", "model": "lr" }
```

- `text` (required): the sentence to analyze. Empty text returns HTTP 400.
- `model` (optional): `"lr"` for Logistic Regression (default). Any other value uses the SVM.

Response shape (values are illustrative):

```json
{
  "sentiment": "positive",
  "confidence": 0.0,
  "scores": { "negative": 0.0, "neutral": 0.0, "positive": 0.0 },
  "word_tags": [ { "word": "Yaar", "lang": "HI" } ],
  "lang_mix": { "hindi": 0, "english": 0, "unknown": 0 },
  "model_used": "Logistic Regression"
}
```

Example call:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Kal meeting hai office mein", "model": "svm"}'
```

### `GET /model_info`

Returns `lr_accuracy`, `svm_accuracy`, `train_size`, `test_size`, `classes` and `features`.

---

## Known limitations

- **Tiny dataset.** 45 sentences cannot cover real Hinglish variety. The notebook's own demo misclassifies `Yaar exam mein bohot kharab gaya, completely failed` as *positive* (confidence 0.42), and several predictions come out with confidence below 0.5.
- **Small language lexicon.** Common Roman-Hindi words that are not in the 45-word list (for example `ye`, `hai`, `thi`) are tagged `UNK`, so the Hindi percentage is an underestimate. Hindi in Devanagari script is not supported.
- **Hard-coded feature count.** `/model_info` always reports `features: 2500`, which is the maximum allowed by the two vectorizers. The actual number is lower (1,345 in the notebook's split).
- **No saved model.** Models are retrained each time the app starts.
- **Development server only.** `app.py` runs Flask with `debug=True`, so it is not suitable for deployment as is.
- **No automated tests.**

## Ideas for improvement

- Train on a larger, real-world Hinglish corpus (tweets, reviews, WhatsApp-style text) and report metrics on a proper held-out set.
- Handle spelling normalisation (`bahut / bohot / bhot`) explicitly, and support Devanagari input.
- Replace the lexicon tagger with a trained language-identification model.
- Compare against a multilingual transformer baseline.
- Save the trained pipeline with `joblib` so the app starts instantly, and add basic tests.

---

## Author

Parul Gupta, B.Tech CSE (AI)

Built with Python, scikit-learn, NLTK and Flask.
