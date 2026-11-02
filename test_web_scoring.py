"""Check the website's scorer (docs/score.js) gives the same results as risk.py.

Needs Node.js.   python test_web_scoring.py
"""
import ensure_env  # noqa: F401  switches to a Python that has the packages if this one doesn't
import json
import random
import subprocess
from dataclasses import asdict

import risk
from label_complaints import ROOT

DESCRIPTIONS = [
    "",
    "brand new chrome hearts hoodie, all sizes available, 1:1 quality, dm me on whatsapp",
    "Patagonia Better Sweater, worn a few times, no stains, measurements in photos. Original receipt available.",
    "UA batch, ships fast. pay with zelle for discount, quick sale need gone today only",
    "Bought from the boutique, authenticated by Entrupy, comes with invoice and dust bag",
    "Selling because it doesn't fit. Seller before me said it's legit but I can't prove it, no returns, avoid lowballers",
    "Vintage Levi's 501, small hole near pocket, priced to sell, bundle for discount",
    "Mirror quality bag, best version on the market, PK batch, wholesale available",
    "Cozy knit, worn twice. Happy to send more pics through the app. Shipping same day.",
    "This seller scammed me before, they claimed the item was authentic but support siding with them, email me",
]
BRANDS = ["", "Chrome Hearts", "Stone Island", "Patagonia", "Carhartt", "Uniqlo", "nike", " Gucci "]


def cases(n=400, seed=7):
    rng = random.Random(seed)
    for _ in range(n):
        yield risk.Listing(
            platform=rng.choice(list(risk.PLATFORMS)),
            brand=rng.choice(BRANDS),
            price=rng.choice([0, 25, 60, 120, 300, 600, 1500]),
            market_price=rng.choice([None, 80, 200, 450, 900]),
            seller_reviews=rng.choice([0, 2, 7, 15, 60, 400]),
            seller_rating=rng.choice([None, 3.9, 4.6, 4.9, 5.0]),
            account_age_days=rng.choice([None, 5, 90, 400, 2000]),
            num_photos=rng.choice([0, 1, 4, 8]),
            stock_photos_only=rng.random() < 0.2,
            shows_tags_or_labels=rng.random() < 0.3,
            multiple_sizes_or_units=rng.random() < 0.15,
            asks_off_platform=rng.random() < 0.15,
            ships_from_unexpected_country=rng.random() < 0.1,
            description=rng.choice(DESCRIPTIONS),
        )


NODE = """
import fs from "node:fs";
import { compileRules, compileModel, score } from "./docs/score.js";
const read = (f) => JSON.parse(fs.readFileSync("docs/data/" + f));
const R = compileRules(read("rules.json")), stats = read("stats.json"), M = compileModel(read("model.json"));
const out = JSON.parse(fs.readFileSync(0)).map((l) => score(l, R, stats, M));
process.stdout.write(JSON.stringify(out));
"""


def main():
    listings = list(cases())
    js = json.loads(subprocess.run(["node", "--input-type=module", "-e", NODE], cwd=ROOT, check=True,
                                   input=json.dumps([asdict(l) for l in listings]),
                                   capture_output=True, text=True).stdout)
    bad = 0
    for l, j in zip(listings, js):
        p = risk.score(l)
        py_reasons = [(w, why) for _, w, why in p.reasons]
        js_reasons = [(r["w"], r["why"]) for r in j["reasons"]]
        same = (abs(p.counterfeit_prob - j["counterfeit"]) < 1e-3 and abs(p.scam_prob - j["scam"]) < 1e-3
                and p.verdict == j["verdict"] and p.advice == j["advice"]
                and sorted(py_reasons) == sorted(js_reasons))
        if not same:
            bad += 1
            if bad <= 3:
                print("MISMATCH", asdict(l))
                print("  py:", round(p.counterfeit_prob, 4), round(p.scam_prob, 4), sorted(py_reasons))
                print("  js:", round(j["counterfeit"], 4), round(j["scam"], 4), sorted(js_reasons))
    print(f"{len(listings) - bad}/{len(listings)} listings score identically in Python and JS")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
