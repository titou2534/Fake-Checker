"""Export the model, platform stats and scoring rules as JSON for the website (docs/).

The website scores listings entirely in the browser, so it needs:
  docs/data/rules.json  - brand lists, platform info and regexes from risk.py
  docs/data/stats.json  - per-platform Reddit complaint stats
  docs/data/model.json  - the TF-IDF + logistic regression text model
Run this after retraining:  python export_web.py
"""
import ensure_env  # noqa: F401  switches to a Python that has the packages if this one doesn't
import json

import joblib

import risk
from label_complaints import ROOT
from train_text_model import PLATFORM_WORDS, WHOLE_WORD_RE

OUT = ROOT / "docs" / "data"


def js_regex(rx):
    # Python's \w is Unicode-aware; spell that out so JS (with the u flag) matches the same characters
    return rx.pattern.replace(r"\w", r"[\p{L}\p{N}_]")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    rules = {
        "base": {"counterfeit": risk.BASE_COUNTERFEIT, "scam": risk.BASE_SCAM},
        "high_risk_brands": sorted(risk.HIGH_RISK_BRANDS),
        "medium_risk_brands": sorted(risk.MEDIUM_RISK_BRANDS),
        "platforms": risk.PLATFORMS,
        "auth_thresholds": risk.AUTH_THRESHOLDS,
        "regex": {name: js_regex(getattr(risk, name)) for name in
                  ["REPLICA_TERMS", "OFFPLATFORM_TERMS", "URGENCY_TERMS", "BULK_TERMS", "PROOF_TERMS"]},
    }
    (OUT / "rules.json").write_text(json.dumps(rules, indent=1, ensure_ascii=False))

    (OUT / "stats.json").write_text(json.dumps(risk.platform_stats(), indent=1))

    model = joblib.load(ROOT / "models" / "text_model.joblib")
    vec, clf = model.named_steps.values()
    terms = sorted(vec.vocabulary_, key=vec.vocabulary_.get)
    export = {
        "intercept": float(clf.intercept_[0]),
        "stop_words": sorted(vec.get_stop_words()),
        "mask": [js_regex(rx) for rx in WHOLE_WORD_RE.values()] + [js_regex(PLATFORM_WORDS)],
        # parallel arrays, indexed like the vectorizer's columns
        "terms": terms,
        "idf": [round(float(x), 4) for x in vec.idf_],
        "coef": [round(float(x), 4) for x in clf.coef_[0]],
    }
    (OUT / "model.json").write_text(json.dumps(export, separators=(",", ":"), ensure_ascii=False))

    for f in sorted(OUT.iterdir()):
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
