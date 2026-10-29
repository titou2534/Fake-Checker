"""Collect real buyer/seller posts about resale platforms from Reddit.

Uses the Arctic Shift public Reddit archive (no API key needed), because
Reddit's own API now blocks unauthenticated requests and X's API is paid.

    python collect_reddit.py --per-sub 1500
"""
import ensure_env  # noqa: F401  switches to a Python that has the packages if this one doesn't
import argparse
import csv
import time
from pathlib import Path

import requests

API = "https://arctic-shift.photon-reddit.com/api/posts/search"
HEADERS = {"User-Agent": "fakecheck-research/0.1 (resale scam research)"}

# subreddit -> platform it is about
SUBREDDITS = {
    "Depop": "depop",
    "vinted": "vinted",
    "Grailed": "grailed",
    "Ebay": "ebay",
    "eBaySellers": "ebay",
    "poshmark": "poshmark",
    "Mercari": "mercari",
    "ThredUp": "thredup",
    "vestiairecollective": "vestiaire",
    "TheRealReal": "therealreal",
}

FIELDS = "id,subreddit,title,selftext,created_utc,score,num_comments"


def fetch_sub(sub, n, pause=1.5):
    """Page backwards in time through a subreddit until n posts are collected."""
    rows, before = [], None
    while len(rows) < n:
        params = {"subreddit": sub, "limit": 100, "fields": FIELDS, "sort": "desc"}
        if before:
            params["before"] = before
        for attempt in range(5):
            try:
                r = requests.get(API, params=params, headers=HEADERS, timeout=60)
                data = r.json().get("data")
                if data is not None:
                    break
            except (requests.RequestException, ValueError):
                pass
            time.sleep(pause * (attempt + 2))  # archive asks clients to slow down
        else:
            print(f"  giving up on r/{sub} after {len(rows)} posts")
            break
        if not data:
            break
        rows.extend(data)
        before = data[-1]["created_utc"]
        time.sleep(pause)
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sub", type=int, default=1500)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "reddit_posts.csv"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    # write to a temp file and only replace the real one at the end, so a failed run keeps the old data
    tmp = out.with_suffix(".partial.csv")
    empty_streak = 0
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "platform", "subreddit", "created_utc",
                                          "score", "num_comments", "title", "selftext"])
        w.writeheader()
        for sub, platform in SUBREDDITS.items():
            posts = fetch_sub(sub, args.per_sub)
            empty_streak = 0 if posts else empty_streak + 1
            if empty_streak >= 2:
                print("archive isn't returning data (it may be down) - stopping early")
                break
            kept = 0
            for p in posts:
                if p["id"] in seen:
                    continue
                seen.add(p["id"])
                text = p.get("selftext") or ""
                if text in ("[removed]", "[deleted]"):
                    text = ""
                w.writerow({"id": p["id"], "platform": platform, "subreddit": sub,
                            "created_utc": p.get("created_utc"), "score": p.get("score"),
                            "num_comments": p.get("num_comments"),
                            "title": p.get("title", ""), "selftext": text})
                kept += 1
            print(f"r/{sub:<20} {kept:>5} posts")
    if empty_streak >= 2 or not seen:
        tmp.unlink()
        raise SystemExit(f"download failed - kept your existing {out}. Try again later.")
    tmp.replace(out)
    print(f"saved {len(seen)} posts -> {out}")


if __name__ == "__main__":
    main()
