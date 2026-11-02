import { compileRules, compileModel, score } from "./score.js";

const $ = (id) => document.getElementById(id);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

// short names for the platform picker; full labels still come from rules.json
const SHORT = { social: "Social / DM", therealreal: "The RealReal", vestiaire: "Vestiaire" };
const ORDER = ["depop", "vinted", "grailed", "ebay", "poshmark", "mercari", "vestiaire", "therealreal", "thredup", "social"];
const DIM = { counterfeit: "fake", scam: "scam", both: "fake + scam" };

const EXAMPLES = {
  rep: {
    platform: "depop", brand: "Chrome Hearts", price: 60, market: 450, reviews: 2, rating: "", age: 12, photos: 2,
    stock: true, tags: false, bulk: true, offplatform: false, ships: false,
    description: "Brand new Chrome Hearts hoodie, all sizes available, 1:1 quality. DM me on WhatsApp for a faster deal.",
  },
  legit: {
    platform: "ebay", brand: "Patagonia", price: 85, market: 100, reviews: 420, rating: 4.9, age: 2000, photos: 8,
    stock: false, tags: true, bulk: false, offplatform: false, ships: false,
    description: "Patagonia Better Sweater, worn a few times, no stains, measurements in photos. Original receipt available.",
  },
};

let R, stats, model = null, modelLoading = null;

async function loadJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

// the text model is ~900 KB, so only fetch it once someone writes a description
function ensureModel() {
  if (model || modelLoading) return;
  modelLoading = loadJSON("data/model.json")
    .then((m) => { model = compileModel(m); render(); })
    .catch(() => { modelLoading = null; });
}

/* ---------- reading the form ---------- */
const num = (id) => { const v = $(id).value.trim(); return v === "" ? null : Number(v); };

function readListing() {
  const p = document.querySelector('input[name="platform"]:checked');
  const price = num("price") ?? 0, market = num("market"), rating = num("rating");
  return {
    platform: p ? p.value : null,
    brand: $("brand").value,
    price,
    market_price: market > 0 ? market : null,
    seller_reviews: num("reviews"),
    seller_rating: rating > 0 ? rating : null,
    account_age_days: num("age"),
    num_photos: num("photos") ?? 0,
    stock_photos_only: $("stock").checked,
    shows_tags_or_labels: $("tags").checked,
    multiple_sizes_or_units: $("bulk").checked,
    asks_off_platform: $("offplatform").checked,
    ships_from_unexpected_country: $("ships").checked,
    description: $("description").value,
  };
}

/* ---------- rendering ---------- */
// Bodoni's lone "1" reads as a bar at display size, so single digits get a faint leading zero
const twoDigits = (n) => (n < 10 ? `<span class="lz">0</span>${n}` : `${n}`);

const tweens = new Map();
function tweenNumber(el, to) {
  const from = Number(el.dataset.v ?? NaN);
  el.dataset.v = to;
  if (reduceMotion || Number.isNaN(from)) { el.innerHTML = twoDigits(to); return; }
  cancelAnimationFrame(tweens.get(el));
  const t0 = performance.now(), dur = 420;
  const step = (t) => {
    const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    el.innerHTML = twoDigits(Math.round(from + (to - from) * e));
    if (k < 1) tweens.set(el, requestAnimationFrame(step));
  };
  tweens.set(el, requestAnimationFrame(step));
}

const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// description with red-flag phrases underlined
function markup(text) {
  const kinds = [["REPLICA_TERMS", "flag"], ["OFFPLATFORM_TERMS", "flag"], ["URGENCY_TERMS", "flag"], ["BULK_TERMS", "flag"], ["PROOF_TERMS", "proof"]];
  const spans = [];
  for (const [name, cls] of kinds) {
    const rx = new RegExp(R.rx[name].source, "giu");
    for (const m of text.matchAll(rx)) if (m[0]) spans.push([m.index, m.index + m[0].length, cls]);
  }
  spans.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
  let out = "", at = 0;
  for (const [s, e, cls] of spans) {
    if (s < at) continue; // overlapping match: keep the first
    out += esc(text.slice(at, s)) + `<mark class="m-${cls}">${esc(text.slice(s, e))}</mark>`;
    at = e;
  }
  return { html: out + esc(text.slice(at)), hits: spans.length };
}

// 2.0 -> "2.0", 0.62 -> "0.62"
const fmtW = (w) => {
  const a = Math.abs(w);
  return a.toFixed(Math.abs(a * 10 - Math.round(a * 10)) < 1e-9 ? 1 : 2);
};

let announceTimer;
function render() {
  if (!R) return;
  const l = readListing();
  const readout = $("readout"), dock = $("dock"), ledger = $("ledger");

  const words = l.description.trim().split(/\s+/).filter(Boolean).length;
  if (words >= 5) ensureModel();

  const mk = $("markup");
  if (l.description.trim()) {
    const { html, hits } = markup(l.description);
    mk.hidden = hits === 0;
    $("markup-text").innerHTML = html;
  } else mk.hidden = true;

  if (!l.platform) {
    readout.dataset.level = dock.dataset.level = "none";
    $("cf").textContent = $("sc").textContent = "–";
    delete $("cf").dataset.v; delete $("sc").dataset.v;
    $("verdict").textContent = "Pick a platform to start.";
    dock.hidden = true; ledger.hidden = true;
    return;
  }

  const r = score(l, R, stats, model);
  const cf = Math.round(r.counterfeit * 100), sc = Math.round(r.scam * 100);
  readout.dataset.level = dock.dataset.level = r.level;
  tweenNumber($("cf"), cf); tweenNumber($("sc"), sc);
  $("dock-cf").innerHTML = twoDigits(cf); $("dock-sc").innerHTML = twoDigits(sc);
  $("verdict").textContent = r.verdict.replace(" - ", ". ");
  dock.hidden = false; ledger.hidden = false;

  const max = Math.max(2.5, ...r.reasons.map((x) => Math.abs(x.w)));
  $("rows").innerHTML = r.reasons.map((x, i) => `
    <li class="row ${x.w > 0 ? "up" : "down"}" data-for="${x.field}" style="--i:${i};--m:${(Math.abs(x.w) / max).toFixed(3)}">
      <span class="w">${x.w > 0 ? "+" : "&minus;"}${fmtW(x.w)}</span>
      <span class="bar" aria-hidden="true"><i></i></span>
      <span class="why">${esc(x.why)}</span>
      <span class="dim">${DIM[x.dim]}</span>
    </li>`).join("");
  $("advice").innerHTML = r.advice.map((t) => `<li>${esc(t)}</li>`).join("");

  clearTimeout(announceTimer);
  announceTimer = setTimeout(() => { $("announce").textContent = `Counterfeit risk ${cf} percent, scam risk ${sc} percent. ${r.verdict}.`; }, 900);
}

// avoid re-running row entrance animations on every keystroke
let renderTimer;
const scheduleRender = () => { clearTimeout(renderTimer); renderTimer = setTimeout(render, 140); };

/* ---------- ledger line -> form field ---------- */
function wirePointing() {
  const rows = $("rows");
  const clear = () => document.querySelectorAll(".pointed").forEach((el) => el.classList.remove("pointed"));
  rows.addEventListener("mouseover", (e) => {
    const row = e.target.closest(".row");
    clear();
    if (row) document.querySelectorAll(`[data-field="${row.dataset.for}"]`).forEach((el) => el.classList.add("pointed"));
  });
  rows.addEventListener("mouseleave", clear);
}

/* ---------- examples ---------- */
function fill(ex) {
  const f = $("dossier");
  if (ex === "clear") {
    f.reset();
    render();
    f.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    return;
  }
  const d = EXAMPLES[ex];
  document.querySelector(`input[name="platform"][value="${d.platform}"]`).checked = true;
  for (const k of ["brand", "price", "market", "reviews", "rating", "age", "photos", "description"]) $(k).value = d[k];
  for (const k of ["stock", "tags", "bulk", "offplatform", "ships"]) $(k).checked = d[k];
  // flash the filled fields one after another
  document.querySelectorAll(".field").forEach((el, i) => {
    setTimeout(() => { el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 500); }, reduceMotion ? 0 : i * 45);
  });
  render();
  const target = matchMedia("(min-width: 1000px)").matches ? $("dossier") : $("readout");
  target.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
}

/* ---------- boot ---------- */
async function boot() {
  [R, stats] = await Promise.all([loadJSON("data/rules.json"), loadJSON("data/stats.json")]);
  R = compileRules(R);

  $("platforms").innerHTML = ORDER.filter((k) => R.platforms[k]).map((k) =>
    `<li><label><input type="radio" name="platform" value="${k}"><span>${SHORT[k] ?? R.platforms[k].label}</span></label></li>`).join("");
  $("brands").innerHTML = [...R.high_risk_brands, ...R.medium_risk_brands]
    .map((b) => `<option value="${b.replace(/\b\w/g, (c) => c.toUpperCase())}">`).join("");

  const form = $("dossier");
  form.addEventListener("input", scheduleRender);
  form.addEventListener("change", render);
  $("description").addEventListener("focus", ensureModel, { once: true });
  document.querySelectorAll(".ex").forEach((b) => b.addEventListener("click", () => fill(b.dataset.ex)));
  wirePointing();
  render();
}

boot().catch((err) => {
  $("verdict").textContent = "Couldn't load the scoring data. Refresh to try again.";
  console.error(err);
});
