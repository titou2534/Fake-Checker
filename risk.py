"""Score a single resale listing for counterfeit risk and scam risk.

Two separate questions, because they need different responses:
  * counterfeit risk - is the *item* likely fake?
  * scam risk        - is the *seller/transaction* likely a scam (no item, phishing, no protection)?

Each signal adds or subtracts log-odds. The weights are hand-set from known
fraud patterns and the Reddit complaint data; they are not learned from
labelled listings (no public dataset of verified fake listings exists).
The text model (train_text_model.py) is the learned component.
"""
import ensure_env  # noqa: F401  switches to a Python that has the packages if this one doesn't
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent

BASE_COUNTERFEIT = -2.4   # ~8% prior that a branded resale listing is fake
BASE_SCAM = -3.0          # ~5% prior that a listing is an outright scam

HIGH_RISK_BRANDS = {
    "nike", "jordan", "yeezy", "adidas yeezy", "supreme", "off-white", "off white", "stone island",
    "chrome hearts", "louis vuitton", "lv", "gucci", "moncler", "canada goose", "arcteryx", "arc'teryx",
    "fear of god", "essentials", "bape", "a bathing ape", "corteiz", "trapstar", "balenciaga", "dior",
    "prada", "chanel", "hermes", "hermès", "rolex", "cartier", "goyard", "celine", "saint laurent", "ysl",
    "new balance", "travis scott", "palm angels", "amiri", "rick owens", "maison margiela", "loewe",
    "miu miu", "bottega veneta", "burberry", "the north face", "north face", "stussy", "sp5der",
    "hellstar", "gallery dept", "denim tears", "vivienne westwood", "ralph lauren", "lululemon",
}
MEDIUM_RISK_BRANDS = {
    "carhartt", "patagonia", "levi's", "levis", "acne studios", "ami", "comme des garcons", "cdg",
    "kith", "palace", "represent", "dr martens", "ugg", "converse", "vans", "tommy hilfiger",
    "lacoste", "kenzo", "versace", "fendi", "givenchy", "valentino", "golden goose", "salomon",
}

# what each platform does for you, used for both risk and advice
PLATFORMS = {
    "ebay":        {"label": "eBay", "auth": "Authenticity Guarantee on eligible sneakers, handbags, watches, jewellery and some streetwear over a price threshold",
                    "protection": "eBay Money Back Guarantee when paid through eBay", "cf": 0.0, "scam": -0.3},
    "grailed":     {"label": "Grailed", "auth": "Moderation team + optional authentication on some listings; no blanket authentication",
                    "protection": "Purchase protection when paid through Grailed checkout", "cf": 0.0, "scam": -0.2},
    "depop":       {"label": "Depop", "auth": "No authentication service",
                    "protection": "Purchase Protection only when paid through Depop Payments", "cf": 0.2, "scam": 0.0},
    "vinted":      {"label": "Vinted", "auth": "Optional Item Verification for designer items in some countries",
                    "protection": "Buyer Protection (fee) when paid in-app; short window to report problems after delivery", "cf": 0.2, "scam": 0.0},
    "poshmark":    {"label": "Poshmark", "auth": "Posh Authenticate on items $500+ (free)",
                    "protection": "Posh Protect when paid through Poshmark", "cf": 0.0, "scam": -0.2},
    "mercari":     {"label": "Mercari", "auth": "Authentication on select brands/categories",
                    "protection": "Buyer protection until you rate the order - don't rate before checking", "cf": 0.1, "scam": 0.0},
    "vestiaire":   {"label": "Vestiaire Collective", "auth": "Items are checked by in-house authenticators",
                    "protection": "Platform-managed returns/refunds for failed authentication", "cf": -1.2, "scam": -0.6},
    "therealreal": {"label": "The RealReal", "auth": "All items authenticated in-house (consignment)",
                    "protection": "Company sells directly", "cf": -1.2, "scam": -1.0},
    "thredup":     {"label": "thredUP", "auth": "Company-sourced inventory; luxury items checked",
                    "protection": "Company sells directly", "cf": -0.8, "scam": -1.0},
    "social":      {"label": "Instagram / Facebook / Telegram / other", "auth": "None",
                    "protection": "None unless you pay with a card or PayPal Goods & Services", "cf": 0.8, "scam": 1.2},
}

AUTH_THRESHOLDS = {"ebay": 100, "poshmark": 500}  # price above which built-in authentication usually applies

REPLICA_TERMS = re.compile(
    r"\b1\s?:\s?1\b|\bUA\b|unauthori[sz]ed authentic|\bmirror\b|\baaa\+?\b|\breps?\b|replica|"
    r"inspired|\bdupe\b|best (?:version|batch)|\bbatch\b|\bPK\b|\bLJR\b|factory (?:direct|seconds)|"
    r"\boem\b|same as (?:retail|original)", re.I)
OFFPLATFORM_TERMS = re.compile(
    r"whats ?app|telegram|text me|email me|dm (?:me )?(?:on|via) (?:insta|ig)|outside (?:the )?app|"
    r"friends (?:and|&) family|\bf ?& ?f\b|zelle|cash ?app|venmo|wire|gift ?card|crypto|bitcoin", re.I)
URGENCY_TERMS = re.compile(r"\burgent|today only|quick sale|need gone|first come|act fast|moving sale", re.I)
BULK_TERMS = re.compile(r"all sizes|any size|multiple (?:sizes|available)|in stock|restock|wholesale|bulk|"
                        r"\d+\s?(?:pcs|pieces|units) available", re.I)
PROOF_TERMS = re.compile(r"receipt|proof of purchase|invoice|authenticated by|legit ?check(?:ed)?|"
                         r"checkcheck|entrupy|legitgrails|order confirmation|bought (?:from|at) (?:the )?(?:store|boutique|snkrs)", re.I)

_text_model = None


def text_model():
    global _text_model
    if _text_model is None:
        path = ROOT / "models" / "text_model.joblib"
        if path.exists():
            import joblib
            _text_model = joblib.load(path)
        else:
            _text_model = False
    return _text_model or None


def platform_stats():
    path = ROOT / "data" / "platform_stats.json"
    return json.loads(path.read_text()) if path.exists() else {}


@dataclass
class Listing:
    platform: str = "depop"
    brand: str = ""
    price: float = 0.0                 # asking price
    market_price: float | None = None  # typical *resale* price for this item in this condition
    seller_reviews: int = 0
    seller_rating: float | None = None  # out of 5
    account_age_days: int | None = None
    num_photos: int = 0
    stock_photos_only: bool = False
    shows_tags_or_labels: bool = False
    multiple_sizes_or_units: bool = False
    asks_off_platform: bool = False    # wants payment/contact outside the app
    ships_from_unexpected_country: bool = False
    description: str = ""


@dataclass
class Result:
    counterfeit_prob: float
    scam_prob: float
    verdict: str
    reasons: list = field(default_factory=list)  # (dimension, weight, text)
    advice: list = field(default_factory=list)


def _sigmoid(z):
    return 1 / (1 + math.exp(-z))


def score(l: Listing) -> Result:
    cf, sc = BASE_COUNTERFEIT, BASE_SCAM
    reasons = []

    def add(dim, w, why):
        nonlocal cf, sc
        if dim in ("counterfeit", "both"):
            cf += w
        if dim in ("scam", "both"):
            sc += w
        reasons.append((dim, w, why))

    plat = PLATFORMS.get(l.platform, PLATFORMS["social"])
    brand = l.brand.strip().lower()
    desc = l.description or ""

    # platform
    if plat["cf"]:
        add("counterfeit", plat["cf"], f"{plat['label']}: {plat['auth']}")
    if plat["scam"]:
        add("scam", plat["scam"], f"{plat['label']}: {plat['protection']}")
    if l.platform in AUTH_THRESHOLDS and l.price >= AUTH_THRESHOLDS[l.platform] and brand in HIGH_RISK_BRANDS | MEDIUM_RISK_BRANDS:
        add("counterfeit", -1.0, f"At this price {plat['label']}'s authentication may apply - check the listing shows the badge")

    stats = platform_stats().get(l.platform)
    if stats:
        rates = [s["counterfeit_rate"] for s in platform_stats().values()]
        mean = sum(rates) / len(rates)
        delta = max(-0.3, min(0.3, (stats["counterfeit_rate"] - mean) * 6))
        if abs(delta) >= 0.05:
            add("counterfeit", round(delta, 2),
                f"Reddit: {stats['counterfeit_rate']:.1%} of r/{plat['label']} posts discuss fakes (avg {mean:.1%})")

    # brand
    if brand in HIGH_RISK_BRANDS:
        add("counterfeit", 0.9, f"'{l.brand}' is one of the most counterfeited brands")
    elif brand in MEDIUM_RISK_BRANDS:
        add("counterfeit", 0.4, f"'{l.brand}' is commonly counterfeited")

    # price
    if l.market_price and l.market_price > 0 and l.price > 0:
        r = l.price / l.market_price
        if r < 0.3:
            add("both", 2.0, f"Price is {r:.0%} of typical resale - far too good to be true")
        elif r < 0.5:
            add("both", 1.3, f"Price is {r:.0%} of typical resale")
        elif r < 0.7:
            add("counterfeit", 0.5, f"Price is {r:.0%} of typical resale - somewhat low")
        elif r <= 1.5:
            add("counterfeit", -0.2, "Price is in the normal resale range")

    # seller history
    if l.seller_reviews == 0:
        add("both", 0.8, "Seller has no reviews")
    elif l.seller_reviews < 10:
        add("both", 0.4, f"Seller has only {l.seller_reviews} reviews")
    elif l.seller_reviews >= 50:
        add("both", -0.6, f"Seller has {l.seller_reviews} reviews")
    if l.seller_rating is not None and l.seller_reviews >= 5:
        if l.seller_rating < 4.5:
            add("both", 0.7, f"Seller rating {l.seller_rating}/5 is below normal (most good sellers are 4.8+)")
        elif l.seller_rating >= 4.9:
            add("both", -0.3, f"Seller rating {l.seller_rating}/5")
    if l.account_age_days is not None:
        if l.account_age_days < 30:
            add("scam", 1.0, "Account is less than a month old")
            add("counterfeit", 0.4, "New account")
        elif l.account_age_days < 180:
            add("both", 0.3, "Account is less than 6 months old")
        elif l.account_age_days > 730:
            add("both", -0.3, "Account is over 2 years old")

    # photos and listing quality
    if l.stock_photos_only:
        add("counterfeit", 1.0, "Only stock/catalogue photos - you can't see the actual item")
        add("scam", 0.8, "Stock photos only: common in non-delivery scams")
    elif 0 < l.num_photos < 3:  # 0 = not entered
        add("counterfeit", 0.5, f"Only {l.num_photos} photo(s)")
    elif l.num_photos >= 6:
        add("counterfeit", -0.3, f"{l.num_photos} real photos")
    if l.shows_tags_or_labels:
        add("counterfeit", -0.5, "Close-ups of tags/labels/serials are shown (lets you legit check)")
    if l.multiple_sizes_or_units or BULK_TERMS.search(desc):
        add("counterfeit", 1.0, "New item in many sizes/units - typical of replica resellers")
    if l.ships_from_unexpected_country:
        add("counterfeit", 0.6, "Ships from somewhere that doesn't match the seller's stated location")

    # transaction
    if l.asks_off_platform or OFFPLATFORM_TERMS.search(desc):
        add("scam", 2.5, "Wants payment/contact outside the platform - this removes buyer protection")
        add("counterfeit", 0.5, "Off-platform requests are common with replica sellers")

    # description text
    if m := REPLICA_TERMS.search(desc):
        add("counterfeit", 2.5, f"Description uses replica-seller language: '{m.group(0)}'")
    if m := URGENCY_TERMS.search(desc):
        add("scam", 0.5, f"Pressure language: '{m.group(0)}'")
    if m := PROOF_TERMS.search(desc):
        add("counterfeit", -0.7, f"Mentions proof of authenticity: '{m.group(0)}'")
    model = text_model()
    if model and len(desc.split()) >= 5:
        from train_text_model import mask  # same preprocessing as training
        p = float(model.predict_proba([mask(desc)])[0, 1])
        # trained on complaint posts, not listings, so a low score isn't evidence of safety: only add risk
        w = round((p - 0.5) * 1.6, 2)
        if w >= 0.1:
            add("both", w, f"Text model: description reads {p:.0%} like posts about scams/fakes")

    cfp, scp = _sigmoid(cf), _sigmoid(sc)
    worst = max(cfp, scp)
    verdict = ("High risk - don't buy without authentication" if worst >= 0.5 else
               "Caution - verify before buying" if worst >= 0.2 else
               "Low risk - normal precautions")
    reasons.sort(key=lambda r: -abs(r[1]))
    return Result(cfp, scp, verdict, reasons, _advice(l, plat, cfp, scp))


def _advice(l, plat, cfp, scp):
    tips = [f"Always pay inside {plat['label']} - {plat['protection']}."]
    if l.asks_off_platform or OFFPLATFORM_TERMS.search(l.description or ""):
        tips.append("Never use PayPal Friends & Family, Zelle, Cash App, gift cards or crypto with a stranger.")
    if cfp >= 0.2:
        tips.append("Ask for tagged photos: the item with a paper showing the seller's username and today's date.")
        tips.append("Ask for close-ups of the inner care tags, size tag, stitching, hardware and any serial/QR code.")
        tips.append("Get a legit check (r/LegitCheck, brand-specific subreddits, or paid services like CheckCheck/LegitGrails).")
    if plat["auth"].lower().startswith(("no ", "none")):
        tips.append(f"{plat['label']} won't authenticate this for you - if it's expensive, consider buying on a platform that does.")
    tips.append("Film yourself opening the parcel; it's the best evidence if you need to open a claim.")
    if l.platform == "mercari":
        tips.append("Don't rate the order on Mercari until you've checked the item - rating ends your protection.")
    return tips


if __name__ == "__main__":
    demo = Listing(platform="depop", brand="Chrome Hearts", price=60, market_price=450,
                   seller_reviews=2, account_age_days=12, num_photos=2, stock_photos_only=True,
                   description="brand new chrome hearts hoodie, all sizes available, 1:1 quality, dm me on whatsapp")
    r = score(demo)
    print(f"counterfeit {r.counterfeit_prob:.0%}  scam {r.scam_prob:.0%}  -> {r.verdict}")
    for dim, w, why in r.reasons:
        print(f"  {w:+.1f} [{dim}] {why}")
