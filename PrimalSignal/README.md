# PrimalSignal

PrimalSignal is a toxicity detection tool built for gaming and online communities. You type in a message, something you'd actually say in a game, and it tells you whether it would get flagged, what category of toxicity it falls under, and how confident the model is.

The idea came from the fact that basic word filters miss a lot. Someone can write around them easily, use leetspeak, or phrase something in a way that sounds innocent word-by-word but is obviously racist or threatening in context. This project tries to close that gap.

---

## What It Does

- Scores any message from 0–100% toxicity risk
- Three verdict states: Clean, Borderline, and Toxic
- Detects obfuscated language like `f*ck`, `f u c k`
- Catches implicit racism and dog-whistle phrasing that word filters miss entirely
- Highlights flagged words directly in your message
- Runs a live moderation log of everything you've analyzed in the session
- Shows a session flag rate so you can see patterns over time

---

## Toxicity Categories

| Label | What It Means |
|---|---|
| Harmful Language | General hostility or harmful tone |
| Extreme Content | Extremely aggressive or hateful language |
| Explicit Content | Sexually explicit or grossly offensive content |
| Threatening | Intent to harm someone |
| Personal Attack | Direct insults or degrading remarks |
| Hate Speech | Hatred targeting race, religion, gender, or ethnicity |

---

## How It Works

The model is a Logistic Regression classifier trained on TF-IDF features with bigrams. Bigrams matter here because phrases like "your kind" or "cotton fields" need to be read as a unit, splitting them into individual words loses the meaning entirely.

On top of the ML model there are two extra layers:

1. **Obfuscation decoder** - normalizes common character substitutions before the model sees the text, so people can't just swap letters to get around the filter

2. **Implicit bias patterns** - a set of phrase-level regex patterns that catch coded racist language the model was never trained to recognize. Things like "didn't know your kind were allowed to be free" score near zero on a word-frequency model but are obviously not okay

The gaming context reducer is also worth mentioning - phrases like "git gud" and "skill issue" get softened slightly so they land in borderline territory rather than being flagged as High severity. That said, if severe language is also present in the same message, the reduction doesn't apply.

---

## Resources

- Python
- Streamlit
- Scikit-learn (Logistic Regression + TF-IDF)
- Plotly
- Regex-based NLP preprocessing

---

## Dataset

Jigsaw Toxic Comment Classification Challenge (Kaggle)

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

Download `train.csv` and place it in the `dataset/` folder before training.

---


## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Model

Only needed if you don't have the `toxic_model.pkl` file, or if you want to retrain from scratch:

```bash
python train_model.py
```

This reads from `dataset/train.csv`, trains the classifier, and saves the model to `models/toxic_model.pkl`. Takes a couple of minutes depending on your machine.

---

## Run the App

```bash
streamlit run app.py
```

It will open automatically in your browser. If it doesn't, go to `http://localhost:8501`.

---

## Notes

The model's main limitation is contextual understanding. It treats text as a bag of words, so it can't fully distinguish "I'm going to destroy you" as competitive trash talk versus a genuine threat. The rule-based layers help bridge this, but there's a ceiling on what's possible without a transformer model.

The real accuracy jump would come from fine-tuning something like DistilBERT on the Jigsaw dataset. That's the logical next step if this ever becomes a production-grade tool.

---

## Possible Next Steps

- Transformer-based model (DistilBERT or similar) for proper contextual understanding
- Discord bot integration for live server moderation
- Real-time multiplayer chat monitoring
- Confidence calibration improvements for short messages
- Export moderation logs to CSV
