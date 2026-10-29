"""Weak-label Reddit posts into complaint categories and summarise per platform.

Each category is a set of regexes. A post can hit several categories.
Outputs:
  data/labeled_posts.csv   - posts + one 0/1 column per category
  data/platform_stats.json - per-platform complaint rates and refund outcomes
"""
import ensure_env  # noqa: F401  switches to a Python that has the packages if this one doesn't
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent

CATEGORIES = {
    # bought something that turned out not to be genuine
    "counterfeit": [
        r"\bfakes?\b", r"counterfeit", r"\breplicas?\b", r"\breps\b", r"\brep (?:seller|site|shoes?|sneakers?|bags?)",r"knock[- ]?offs?",
        r"not (?:authentic|legit|real|genuine)", r"inauthentic", r"\bbootleg",
        r"failed authentication", r"legit ?check", r"\bLC\b", r"\bdupes?\b",
    ],
    # paid and got nothing / got an empty box / seller vanished
    "non_delivery": [
        r"scamm?(?:ed|er|ers)\b", r"never (?:arrived|received|came|shipped)",
        r"(?:didn'?t|did not|hasn'?t|has not) (?:arrive|ship|send)",
        r"empty (?:box|package|parcel)", r"fake tracking", r"\bghost(?:ed|ing)\b",
        r"marked (?:as )?delivered",
    ],
    # pushed to pay or talk outside the platform (kills buyer protection)
    "off_platform": [
        r"friends (?:and|&) family", r"\bf ?& ?f\b", r"\bzelle\b", r"cash ?app", r"\bvenmo\b",
        r"whats ?app", r"telegram", r"outside (?:of )?(?:the )?(?:app|platform|site)",
        r"off[- ]platform", r"(?:text|email) me", r"wire transfer", r"gift ?cards?",
    ],
    # phishing / fake payment emails / account takeover
    "account_fraud": [
        r"phishing", r"hack(?:ed|er)", r"verification code", r"fake (?:email|link|website)",
        r"suspicious link", r"(?:qr|payment) link", r"account (?:was )?(?:taken|stolen|compromised)",
    ],
    # item arrived but was wrong / damaged / misrepresented
    "not_as_described": [
        r"not as described", r"\bsnad\b", r"\bINAD\b", r"(?:wrong|different) (?:item|size|colou?r)",
        r"stain(?:ed|s)?\b", r"\bholes?\b", r"damaged", r"misrepresent",
    ],
    # buyer scamming a seller (returns swapped item, false claims)
    "buyer_fraud": [
        r"return(?:ed)? (?:a |an )?(?:fake|different|wrong|empty)", r"swapped",
        r"(?:buyer|they) claim(?:ed|s)? (?:it|the item) (?:was )?(?:fake|never)",
        r"chargeback", r"return fraud", r"wardrob(?:ing|e fraud)",
    ],
}

# keywords that tell us how a dispute ended
RESOLVED_GOOD = [r"got (?:my |a |full )?(?:money|refund)", r"refunded", r"sided with me",
                 r"case (?:was )?(?:closed )?in my favou?r", r"got it back"]
RESOLVED_BAD = [r"(?:denied|refused|rejected) (?:my )?(?:refund|claim|case)", r"sided with (?:the )?(?:seller|buyer|scammer)",
                r"no (?:buyer )?protection", r"(?:won'?t|will not|refuse to) refund", r"lost (?:my )?(?:money|\$)"]

RISK_CATS = ["counterfeit", "non_delivery", "off_platform", "account_fraud"]


def compile_all(patterns):
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.I)


CAT_RE = {c: compile_all(p) for c, p in CATEGORIES.items()}
GOOD_RE, BAD_RE = compile_all(RESOLVED_GOOD), compile_all(RESOLVED_BAD)


def label(df):
    text = (df["title"].fillna("") + " \n " + df["selftext"].fillna(""))
    df = df.assign(text=text)
    for c, rx in CAT_RE.items():
        df[c] = text.str.contains(rx).astype(int)
    df["any_complaint"] = df[list(CATEGORIES)].max(axis=1)
    df["risk_post"] = df[RISK_CATS].max(axis=1)
    df["resolved_good"] = text.str.contains(GOOD_RE).astype(int)
    df["resolved_bad"] = text.str.contains(BAD_RE).astype(int)
    return df


def platform_stats(df):
    stats = {}
    for plat, g in df.groupby("platform"):
        risky = g[g["risk_post"] == 1]
        good, bad = int(risky["resolved_good"].sum()), int(risky["resolved_bad"].sum())
        stats[plat] = {
            "posts": int(len(g)),
            **{f"{c}_rate": round(float(g[c].mean()), 4) for c in list(CATEGORIES) + ["risk_post"]},
            "refund_win_rate": round(good / (good + bad), 3) if good + bad >= 5 else None,
            "refund_outcomes_seen": good + bad,
        }
    return stats


def main():
    df = pd.read_csv(ROOT / "data" / "reddit_posts.csv")
    df = df.drop_duplicates("id")
    df = label(df)
    df.drop(columns="text").to_csv(ROOT / "data" / "labeled_posts.csv", index=False)
    stats = platform_stats(df)
    with open(ROOT / "data" / "platform_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    table = pd.DataFrame(stats).T[["posts", "risk_post_rate", "counterfeit_rate", "non_delivery_rate",
                                   "off_platform_rate", "account_fraud_rate", "refund_win_rate"]]
    print(table.sort_values("risk_post_rate", ascending=False).to_string())


if __name__ == "__main__":
    main()
