import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# Load the Jigsaw dataset
print("Loading dataset...")
data = pd.read_csv("dataset/train.csv")

data = data[["comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
data = data.rename(columns={"comment_text": "message"})

label_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

messages = data["message"]
labels   = data[label_columns]

print("Splitting into train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    messages, labels, test_size=0.2, random_state=42
)

# Why this is better than the old Naive Bayes setup:
#
# 1. TF-IDF now uses bigrams (ngram_range=(1,2)) — this lets it learn
#    "your kind", "cotton field", "kill yourself" as single features,
#    which is crucial for catching implicit racism and multi-word slurs.
#
# 2. sublinear_tf=True dampens the effect of a word appearing 100 times
#    vs 10 times — more realistic for real chat messages.
#
# 3. LogisticRegression instead of Naive Bayes — LR learns negative weights
#    too, so it can "unlearn" false positives (e.g. "kill it" in gaming
#    context). NB only sees positive co-occurrences.
#
# 4. class_weight="balanced" compensates for the heavy class imbalance
#    in the dataset (most comments are non-toxic), so the model doesn't
#    just predict "safe" for everything ambiguous.

print("Building pipeline...")
pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            max_features=50000,
            ngram_range=(1, 2),       # captures word pairs, not just single words
            sublinear_tf=True,        # log-scale term frequency
            strip_accents="unicode",
            analyzer="word",
        )
    ),
    (
        "classifier",
        OneVsRestClassifier(
            LogisticRegression(
                C=1.0,
                max_iter=200,
                solver="lbfgs",
                class_weight="balanced",   # handles class imbalance
            )
        )
    )
])

print("Training (this takes a couple of minutes)...")
pipeline.fit(X_train, y_train)

print("Evaluating...")
predictions = pipeline.predict(X_test)
print(classification_report(y_test, predictions, target_names=label_columns))

print("Saving model...")
pickle.dump(pipeline, open("models/toxic_model.pkl", "wb"))

print("Done. New model saved to models/toxic_model.pkl")
