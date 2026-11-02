// Browser port of risk.py. Keep the two in sync; test_web_scoring.py checks they agree.
// Rules, stats and the text model come from export_web.py (docs/data/*.json).

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const round2 = (x) => Math.round(x * 100) / 100;
// Python formats with round-half-to-even (12.5% -> 12%); match it
const roundHalfEven = (v) => {
  const f = Math.floor(v);
  return Math.abs(v - f - 0.5) < 1e-9 ? (f % 2 === 0 ? f : f + 1) : Math.round(v);
};
const pct = (x) => `${roundHalfEven(x * 100)}%`;
const pct1 = (x) => `${(x * 100).toFixed(1)}%`;
const pyFloat = (x) => (Number.isInteger(x) ? x.toFixed(1) : `${x}`); // Python prints 5.0, JS prints 5

export function compileRules(rules) {
  const rx = {};
  for (const [name, src] of Object.entries(rules.regex)) rx[name] = new RegExp(src, "iu");
  return { ...rules, rx, high: new Set(rules.high_risk_brands), medium: new Set(rules.medium_risk_brands) };
}

export function compileModel(model) {
  const index = new Map(model.terms.map((t, i) => [t, i]));
  return {
    ...model,
    index,
    stop: new Set(model.stop_words),
    maskRx: model.mask.map((src) => new RegExp(src, "giu")),
  };
}

// same preprocessing as train_text_model.mask + TfidfVectorizer + LogisticRegression
export function textProb(model, text) {
  for (const rx of model.maskRx) text = text.replace(rx, " ");
  const words = (text.toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) || []).filter((w) => !model.stop.has(w));
  const tf = new Map();
  const bump = (t) => tf.set(t, (tf.get(t) || 0) + 1);
  words.forEach((w, i) => {
    bump(w);
    if (i + 1 < words.length) bump(`${w} ${words[i + 1]}`);
  });
  let dot = 0, norm = 0;
  for (const [t, n] of tf) {
    const i = model.index.get(t);
    if (i === undefined) continue;
    const v = (1 + Math.log(n)) * model.idf[i];
    dot += v * model.coef[i];
    norm += v * v;
  }
  return sigmoid((norm ? dot / Math.sqrt(norm) : 0) + model.intercept);
}

// field = which form input a reason came from, so the UI can point at it
export function score(l, R, stats, model) {
  let cf = R.base.counterfeit, sc = R.base.scam;
  const reasons = [];
  const add = (dim, w, why, field) => {
    if (dim === "counterfeit" || dim === "both") cf += w;
    if (dim === "scam" || dim === "both") sc += w;
    reasons.push({ dim, w, why, field });
  };

  const plat = R.platforms[l.platform] || R.platforms.social;
  const brand = l.brand.trim().toLowerCase();
  const desc = l.description || "";

  if (plat.cf) add("counterfeit", plat.cf, `${plat.label}: ${plat.auth}`, "platform");
  if (plat.scam) add("scam", plat.scam, `${plat.label}: ${plat.protection}`, "platform");
  if (l.platform in R.auth_thresholds && l.price >= R.auth_thresholds[l.platform] && (R.high.has(brand) || R.medium.has(brand)))
    add("counterfeit", -1.0, `At this price ${plat.label}'s authentication may apply - check the listing shows the badge`, "price");

  const st = stats[l.platform];
  if (st) {
    const rates = Object.values(stats).map((s) => s.counterfeit_rate);
    const mean = rates.reduce((a, b) => a + b, 0) / rates.length;
    const delta = Math.max(-0.3, Math.min(0.3, (st.counterfeit_rate - mean) * 6));
    if (Math.abs(delta) >= 0.05)
      add("counterfeit", round2(delta), `Reddit: ${pct1(st.counterfeit_rate)} of r/${plat.label} posts discuss fakes (avg ${pct1(mean)})`, "platform");
  }

  if (R.high.has(brand)) add("counterfeit", 0.9, `'${l.brand.trim()}' is one of the most counterfeited brands`, "brand");
  else if (R.medium.has(brand)) add("counterfeit", 0.4, `'${l.brand.trim()}' is commonly counterfeited`, "brand");

  if (l.market_price && l.market_price > 0 && l.price > 0) {
    const r = l.price / l.market_price;
    if (r < 0.3) add("both", 2.0, `Price is ${pct(r)} of typical resale - far too good to be true`, "price");
    else if (r < 0.5) add("both", 1.3, `Price is ${pct(r)} of typical resale`, "price");
    else if (r < 0.7) add("counterfeit", 0.5, `Price is ${pct(r)} of typical resale - somewhat low`, "price");
    else if (r <= 1.5) add("counterfeit", -0.2, "Price is in the normal resale range", "price");
  }

  // the website allows a blank (null) review count = unknown; risk.py always has a number
  if (l.seller_reviews === 0) add("both", 0.8, "Seller has no reviews", "reviews");
  else if (l.seller_reviews != null && l.seller_reviews < 10) add("both", 0.4, `Seller has only ${l.seller_reviews} reviews`, "reviews");
  else if (l.seller_reviews >= 50) add("both", -0.6, `Seller has ${l.seller_reviews} reviews`, "reviews");
  if (l.seller_rating != null && l.seller_reviews >= 5) {
    if (l.seller_rating < 4.5) add("both", 0.7, `Seller rating ${pyFloat(l.seller_rating)}/5 is below normal (most good sellers are 4.8+)`, "rating");
    else if (l.seller_rating >= 4.9) add("both", -0.3, `Seller rating ${pyFloat(l.seller_rating)}/5`, "rating");
  }
  if (l.account_age_days != null) {
    if (l.account_age_days < 30) {
      add("scam", 1.0, "Account is less than a month old", "age");
      add("counterfeit", 0.4, "New account", "age");
    } else if (l.account_age_days < 180) add("both", 0.3, "Account is less than 6 months old", "age");
    else if (l.account_age_days > 730) add("both", -0.3, "Account is over 2 years old", "age");
  }

  if (l.stock_photos_only) {
    add("counterfeit", 1.0, "Only stock/catalogue photos - you can't see the actual item", "stock");
    add("scam", 0.8, "Stock photos only: common in non-delivery scams", "stock");
  } else if (l.num_photos > 0 && l.num_photos < 3) add("counterfeit", 0.5, `Only ${l.num_photos} photo(s)`, "photos");
  else if (l.num_photos >= 6) add("counterfeit", -0.3, `${l.num_photos} real photos`, "photos");
  if (l.shows_tags_or_labels) add("counterfeit", -0.5, "Close-ups of tags/labels/serials are shown (lets you legit check)", "tags");
  if (l.multiple_sizes_or_units || R.rx.BULK_TERMS.test(desc))
    add("counterfeit", 1.0, "New item in many sizes/units - typical of replica resellers", l.multiple_sizes_or_units ? "bulk" : "description");
  if (l.ships_from_unexpected_country) add("counterfeit", 0.6, "Ships from somewhere that doesn't match the seller's stated location", "ships");

  const offField = l.asks_off_platform ? "offplatform" : "description";
  if (l.asks_off_platform || R.rx.OFFPLATFORM_TERMS.test(desc)) {
    add("scam", 2.5, "Wants payment/contact outside the platform - this removes buyer protection", offField);
    add("counterfeit", 0.5, "Off-platform requests are common with replica sellers", offField);
  }

  let m;
  if ((m = desc.match(R.rx.REPLICA_TERMS))) add("counterfeit", 2.5, `Description uses replica-seller language: '${m[0]}'`, "description");
  if ((m = desc.match(R.rx.URGENCY_TERMS))) add("scam", 0.5, `Pressure language: '${m[0]}'`, "description");
  if ((m = desc.match(R.rx.PROOF_TERMS))) add("counterfeit", -0.7, `Mentions proof of authenticity: '${m[0]}'`, "description");
  if (model && desc.trim().split(/\s+/).length >= 5) {
    const p = textProb(model, desc);
    const w = round2((p - 0.5) * 1.6);
    // trained on complaint posts, not listings, so a low score isn't evidence of safety: only add risk
    if (w >= 0.1) add("both", w, `Text model: description reads ${pct(p)} like posts about scams/fakes`, "description");
  }

  const cfp = sigmoid(cf), scp = sigmoid(sc);
  const worst = Math.max(cfp, scp);
  const level = worst >= 0.5 ? "high" : worst >= 0.2 ? "caution" : "low";
  const verdict = { high: "High risk - don't buy without authentication", caution: "Caution - verify before buying", low: "Low risk - normal precautions" }[level];
  reasons.sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
  return { counterfeit: cfp, scam: scp, level, verdict, reasons, advice: advice(l, plat, cfp, R) };
}

function advice(l, plat, cfp, R) {
  const tips = [`Always pay inside ${plat.label} - ${plat.protection}.`];
  if (l.asks_off_platform || R.rx.OFFPLATFORM_TERMS.test(l.description || ""))
    tips.push("Never use PayPal Friends & Family, Zelle, Cash App, gift cards or crypto with a stranger.");
  if (cfp >= 0.2) {
    tips.push("Ask for tagged photos: the item with a paper showing the seller's username and today's date.");
    tips.push("Ask for close-ups of the inner care tags, size tag, stitching, hardware and any serial/QR code.");
    tips.push("Get a legit check (r/LegitCheck, brand-specific subreddits, or paid services like CheckCheck/LegitGrails).");
  }
  if (/^(no |none)/i.test(plat.auth))
    tips.push(`${plat.label} won't authenticate this for you - if it's expensive, consider buying on a platform that does.`);
  tips.push("Film yourself opening the parcel; it's the best evidence if you need to open a claim.");
  if (l.platform === "mercari") tips.push("Don't rate the order on Mercari until you've checked the item - rating ends your protection.");
  return tips;
}
