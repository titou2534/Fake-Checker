"""Web app:  streamlit run app.py   (or just `python app.py` / VS Code's Run button)"""
import sys

import ensure_env  # noqa: F401  switches to a Python that has the packages if this one doesn't
import pandas as pd
import streamlit as st
from streamlit import runtime

if __name__ == "__main__" and not runtime.exists():
    # started with plain `python app.py`: relaunch through Streamlit
    from streamlit.web import cli
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(cli.main())

from risk import PLATFORMS, Listing, platform_stats, score

st.set_page_config(page_title="Fake Checker", page_icon="🔍", layout="centered")
st.title("Fake Checker")
st.caption("Is that second-hand listing fake, or a scam? Scores counterfeit and scam risk "
           "from the listing details, using patterns mined from Reddit complaints.")

check, report = st.tabs(["Check a listing", "Platform report"])

with check:
    keys = list(PLATFORMS)
    c1, c2 = st.columns(2)
    platform = c1.selectbox("Where is it listed?", keys, format_func=lambda k: PLATFORMS[k]["label"])
    brand = c2.text_input("Brand", placeholder="e.g. Stone Island")
    c1, c2 = st.columns(2)
    price = c1.number_input("Asking price", min_value=0.0, value=0.0, step=5.0)
    market = c2.number_input("Typical resale price (check sold listings)", min_value=0.0, value=0.0, step=5.0,
                             help="Search the same item in sold/completed listings and take a typical price.")
    c1, c2, c3 = st.columns(3)
    reviews = c1.number_input("Seller reviews", min_value=0, value=0, step=1)
    rating = c2.number_input("Seller rating (/5)", min_value=0.0, max_value=5.0, value=0.0, step=0.1,
                             help="Leave at 0 if unknown")
    age = c3.number_input("Account age (days)", min_value=0, value=0, step=30, help="Leave at 0 if unknown")
    c1, c2 = st.columns(2)
    photos = c1.number_input("Number of photos", min_value=0, value=0, step=1)
    stock = c2.checkbox("Only stock / catalogue photos")
    tags = c2.checkbox("Shows close-ups of tags / labels / serial")
    bulk = c1.checkbox("Same new item in many sizes or units")
    off = c1.checkbox("Seller wants to pay or chat outside the app")
    ships = c2.checkbox("Ships from somewhere unexpected")
    desc = st.text_area("Paste the listing description and/or seller messages", height=120)

    if st.button("Check listing", type="primary"):
        r = score(Listing(
            platform=platform, brand=brand, price=price, market_price=market or None,
            seller_reviews=int(reviews), seller_rating=rating or None, account_age_days=int(age) or None,
            num_photos=int(photos), stock_photos_only=stock, shows_tags_or_labels=tags,
            multiple_sizes_or_units=bulk, asks_off_platform=off, ships_from_unexpected_country=ships,
            description=desc,
        ))
        worst = max(r.counterfeit_prob, r.scam_prob)
        (st.error if worst >= 0.5 else st.warning if worst >= 0.2 else st.success)(r.verdict)
        m1, m2 = st.columns(2)
        m1.metric("Counterfeit risk", f"{r.counterfeit_prob:.0%}")
        m2.metric("Scam risk", f"{r.scam_prob:.0%}")

        st.subheader("Why")
        for dim, w, why in r.reasons:
            icon = "🔴" if w >= 1 else "🟠" if w > 0 else "🟢"
            st.write(f"{icon} **{w:+.1f}** · {why}  \n<small>{dim}</small>", unsafe_allow_html=True)
        st.subheader("What to do")
        for tip in r.advice:
            st.write(f"- {tip}")
        st.caption("Risk scores are estimates, not proof. A low score doesn't guarantee an item is authentic.")

with report:
    stats = platform_stats()
    if not stats:
        st.info("Run collect_reddit.py and label_complaints.py to build the platform report.")
    else:
        df = pd.DataFrame(stats).T
        df.index = [PLATFORMS.get(k, {"label": k})["label"] for k in df.index]
        st.write(f"Based on **{int(df['posts'].sum()):,}** Reddit posts from each platform's subreddit. "
                 "Rates are the share of posts that mention each problem - they reflect what people "
                 "complain about, not the true fraud rate.")
        cols = {"counterfeit_rate": "Fakes", "non_delivery_rate": "Non-delivery",
                "off_platform_rate": "Off-platform", "account_fraud_rate": "Phishing/hacks",
                "not_as_described_rate": "Not as described", "buyer_fraud_rate": "Buyer fraud"}
        chart = df[list(cols)].rename(columns=cols).astype(float).mul(100)
        st.bar_chart(chart, horizontal=True, stack=False, y_label="% of posts")
        table = df[["posts", "risk_post_rate", "refund_win_rate", "refund_outcomes_seen"]].rename(columns={
            "posts": "Posts", "risk_post_rate": "Scam/fake posts", "refund_win_rate": "Refund won (when mentioned)",
            "refund_outcomes_seen": "Outcomes seen"})
        st.dataframe(table.sort_values("Scam/fake posts", ascending=False), width="stretch",
                     column_config={"Scam/fake posts": st.column_config.NumberColumn(format="percent"),
                                    "Refund won (when mentioned)": st.column_config.NumberColumn(format="percent")})
        st.subheader("What each platform does for you")
        for k, p in PLATFORMS.items():
            st.write(f"**{p['label']}** - {p['auth']}. {p['protection']}.")
        st.caption("Platform policies change; check the current terms before relying on them.")
