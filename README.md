# Fake Checker

Checks a second-hand clothing listing (Grailed, eBay, Depop, Vinted, Poshmark, Mercari,
Vestiaire, The RealReal, thredUP, or social media) and gives two scores:

- **Counterfeit risk**: how likely the item is fake
- **Scam risk**: how likely the seller or the deal is a scam (no item sent, phishing, payment outside the app)

It also lists the reasons for each score and tells you what to do next.

## Quick start

The repo ships with the trained model and the platform stats, so this is enough to use the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

To rebuild everything from fresh Reddit data (the raw posts aren't stored in the repo):

```bash
python collect_reddit.py --per-sub 1500   # ~15k real posts from platform subreddits (takes a few minutes)
python label_complaints.py                # tag complaints + per-platform stats
python train_text_model.py                # train the text model
streamlit run app.py                      # open the web app
```

## How it works

| Step | File | What it does |
|---|---|---|
| Data | `collect_reddit.py` | Downloads posts from r/Depop, r/vinted, r/Grailed, r/Ebay, r/poshmark, r/Mercari and others through the Arctic Shift Reddit archive. Reddit's own API blocks requests without a login, and X's API is paid. |
| Labels | `label_complaints.py` | Uses regex rules to tag each post as counterfeit, non-delivery, off-platform payment, phishing, not as described, or buyer fraud. It then works out complaint rates per platform, and how often people say they got their refund. |
| ML | `train_text_model.py` | Trains TF-IDF + logistic regression to spot scam or counterfeit context in text. The labelling keywords are removed before training, so the model learns the language around them instead of copying the rules. |
| Scoring | `risk.py` | Adds up log-odds from each signal: price compared to typical resale, seller history, photos, brand, what the platform protects, replica wording (`1:1`, `UA`, `best batch`), requests to pay or chat off-platform, and the text model. |
| App | `app.py` | Streamlit app with two tabs: check a listing, and a report comparing platforms. |

You can also use the scorer from Python:

```python
from risk import Listing, score
r = score(Listing(platform="grailed", brand="Rick Owens", price=180, market_price=600,
                  seller_reviews=3, num_photos=2, description="DM me on IG for more pics"))
print(r.counterfeit_prob, r.scam_prob, r.verdict, r.reasons)
```

## Limitations

- **Nobody publishes a labelled dataset of confirmed fake listings.** That's why the weights in
  `risk.py` are set by hand from known fraud patterns, not learned. The scores rank listings
  sensibly, but they aren't calibrated probabilities.
- The Reddit labels come from keyword rules, so they're noisy. Complaint rates show what people
  post about, and subreddit culture affects that. They are not true fraud rates.
- The text model was trained on Reddit posts, not on listing descriptions. Treat it as one
  signal among several.
- Photos matter most for spotting fakes, and this tool doesn't look at images.

## Ideas for next steps

1. **Real labels:** record each checked listing in a CSV along with how it turned out
   (legit check result, refund, item received). With a few hundred rows, fit a logistic
   regression on the same features as `risk.py` to replace the hand-set weights.
2. **Images:** fine-tune an image model on legit-check posts (r/LegitCheck, r/Sneakers
   "LC" threads), using the community's majority verdict as the label.
3. **Listing URLs:** parse a pasted listing URL to fill in the form automatically. Check
   each site's terms first.
