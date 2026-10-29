"""Train a text model that spots scam / counterfeit context in free text.

Labels come from label_complaints.py (risk_post = counterfeit, non-delivery,
off-platform or account-fraud). Those labels are keyword-based, so before
training we *mask out every labelling keyword*: otherwise the model would just
memorise the regexes. What it learns instead is the surrounding language that
co-occurs with scams ("dm me", "no returns", "brand new in box all sizes", ...),
which is what makes it useful on text the regexes miss.

    python train_text_model.py
"""
import ensure_env  # noqa: F401  switches to a Python that has the packages if this one doesn't
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from label_complaints import CAT_RE, RISK_CATS, ROOT

PLATFORM_WORDS = re.compile(r"\b(?:depop|vinted|grailed|ebay|poshmark|posh|mercari|thredup|"
                            r"vestiaire|the ?real ?real|trr)\b", re.I)


WHOLE_WORD_RE = {c: re.compile(rf"\w*(?:{CAT_RE[c].pattern})\w*", re.I) for c in RISK_CATS}


def mask(text):
    # drop the whole word around each match so no fragments ("shipped" -> "ped") remain
    for c in RISK_CATS:
        text = WHOLE_WORD_RE[c].sub(" ", text)
    return PLATFORM_WORDS.sub(" ", text)


def main():
    df = pd.read_csv(ROOT / "data" / "labeled_posts.csv")
    df["text"] = (df["title"].fillna("") + " \n " + df["selftext"].fillna("")).map(mask)
    df = df[df["text"].str.split().str.len() >= 5]
    y = df["risk_post"].values
    print(f"{len(df)} posts, {y.mean():.1%} labelled risky")

    X_tr, X_te, y_tr, y_te = train_test_split(df["text"], y, test_size=0.2,
                                              stratify=y, random_state=42)
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True,
                        stop_words="english"),
        LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000),
    )
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    print(f"held-out ROC AUC (keywords masked): {roc_auc_score(y_te, p):.3f}")
    print(classification_report(y_te, p > 0.5, target_names=["normal", "risky"], digits=3))

    vec, clf = model.named_steps.values()
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    print("top risky phrases :", ", ".join(names[order[-30:]][::-1]))
    print("top normal phrases:", ", ".join(names[order[:15]]))

    model.fit(df["text"], y)  # refit on everything for the shipped model
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "text_model.joblib")
    print("saved models/text_model.joblib")


if __name__ == "__main__":
    main()
