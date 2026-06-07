"""
05_generate_figures.py — Regenerate all output figures from pooled_events.csv.

Usage:
    python code/05_generate_figures.py
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent.parent
PROC    = ROOT / "data" / "processed"
RAW     = ROOT / "data" / "raw"
OUT     = ROOT / "output"
OUT.mkdir(exist_ok=True)

WHALE_THRESHOLD = 5_000

# ── Load & filter data ─────────────────────────────────────────────────────────
print("Loading pooled_events.csv...")
df = pd.read_csv(PROC / "pooled_events.csv", parse_dates=["timestamp"], low_memory=False)
n_markets_total = df["market_label"].nunique()
print(f"  {len(df):,} rows, {n_markets_total} markets")

df = df[df["price_change"].notna()].copy()
if "is_convergence_period" in df.columns:
    df = df[~df["is_convergence_period"]].copy()

if "abs_price_change" not in df.columns:
    df["abs_price_change"] = df["price_change"].abs()
df["is_whale"] = df["position_usd"] >= WHALE_THRESHOLD

# Remove markets with constant price data (broken price history)
mkt_std = df.groupby("market_label")["abs_price_change"].std()
broken_mkts = mkt_std[mkt_std < 0.001].index
df = df[~df["market_label"].isin(broken_mkts)].copy()

# Build id_to_q early so we can use it for the Fed filter
id_to_q: dict[str, str] = {}
for p in RAW.iterdir():
    mj = p / "market.json"
    if mj.exists():
        try:
            m = json.loads(mj.read_text())
            q = m.get("question", "")
            if q: id_to_q[p.name] = q
        except Exception:
            pass

# Remove Fed Policy markets (no meaningful whale activity)
FED_KEYWORDS = ["fed ", "fomc", "interest rate", "rate cut", "rate hike",
                "basis point", "bps", "federal reserve"]
fed_mkts = [lbl for lbl in df["market_label"].unique()
            if any(k in id_to_q.get(str(lbl), "").lower() for k in FED_KEYWORDS)]
df = df[~df["market_label"].isin(fed_mkts)].copy()
print(f"  Removed {len(fed_mkts)} Fed Policy markets")

n_markets_clean = df["market_label"].nunique()
n_whale  = int(df["is_whale"].sum())
n_retail = int((~df["is_whale"]).sum())
n_whale_markets = int((df.groupby("market_label")["is_whale"].sum() > 0).sum())

print(f"  After filtering: {len(df):,} events from {n_markets_clean} markets")
print(f"  Whale events={n_whale:,}  Retail events={n_retail:,}  Whale markets={n_whale_markets}")

# ── Market name lookup (id_to_q already populated above) ──────────────────────
for p in RAW.iterdir():
    mj = p / "market.json"
    if mj.exists():
        try:
            m = json.loads(mj.read_text())
            q = m.get("question", "")
            if q:
                id_to_q[p.name] = q
        except Exception:
            pass

def market_name(label, maxlen=36) -> str:
    q = id_to_q.get(str(label), str(label))
    return q[:maxlen-1] + "…" if len(q) > maxlen else q

# ──────────────────────────────────────────────────────────────────────────────
# FIG 1 — Price trajectories (6 representative markets)
# ──────────────────────────────────────────────────────────────────────────────
print("\nFig 1: price trajectories...")

candidates = (
    df.groupby("market_label")
    .agg(vol=("market_volume","first"), n_whale=("is_whale","sum"),
         n=("price_change","count"), std_price=("price_before","std"))
    .query("n_whale >= 5 and n >= 100 and std_price >= 0.02")
    .sort_values("vol", ascending=False)
)
top6 = candidates.head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, label in zip(axes.flat, top6):
    sub = df[df["market_label"] == label].sort_values("timestamp")
    if "price_before" not in sub.columns or sub["price_before"].isna().all():
        ax.set_visible(False); continue
    ax.plot(sub["timestamp"], sub["price_before"], alpha=0.45, lw=0.9, color="steelblue")
    whale_sub = sub[sub["is_whale"]]
    ax.scatter(whale_sub["timestamp"], whale_sub["price_before"],
               color="red", s=20, zorder=5, alpha=0.8)
    ax.axhline(0.1, color="gray", lw=0.6, ls="--", alpha=0.4)
    ax.axhline(0.9, color="gray", lw=0.6, ls="--", alpha=0.4)
    ax.set_ylim(-0.05, 1.05)
    vol = sub["market_volume"].iloc[0] if "market_volume" in sub.columns else 0
    ax.set_title(f"{market_name(label)}\n(${vol:,.0f})", fontsize=8, pad=4)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("YES probability", fontsize=8)

import matplotlib.lines as mlines
line = mlines.Line2D([], [], color='steelblue', linewidth=1.5, label='YES price')
dot  = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=6, label='Whale comment')
fig.legend(handles=[line, dot], loc="lower center", ncol=2, fontsize=10, framealpha=0.8)
plt.suptitle("YES-Token Price Trajectories — 6 Representative Markets\n"
             "Red dots = whale comment events", fontsize=11, y=1.01)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT / "fig1_price_trajectories.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Saved fig1")

# ──────────────────────────────────────────────────────────────────────────────
# FIG 2 — Top 5 markets by volume: whale vs retail absolute |Δp|
# ──────────────────────────────────────────────────────────────────────────────
print("Fig 2: top 5 markets, absolute price change...")

stats_rows = []
for label, sub in df.groupby("market_label"):
    whale  = sub[sub["is_whale"]]["abs_price_change"].dropna()
    retail = sub[~sub["is_whale"]]["abs_price_change"].dropna()
    if len(whale) < 3 or len(retail) < 10:
        continue
    vol = float(sub["market_volume"].iloc[0]) if "market_volume" in sub.columns else 0
    stats_rows.append({"label": label, "volume": vol,
                       "whale_mean": whale.mean(), "whale_se": whale.sem(),
                       "retail_mean": retail.mean(), "retail_se": retail.sem(),
                       "n_whale": len(whale)})

mdf = pd.DataFrame(stats_rows).sort_values("volume", ascending=False)
top5 = mdf.head(5)

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(top5)); w = 0.35
ax.bar(x - w/2, top5["whale_mean"],  w, label="Whale",  color="steelblue",
       yerr=top5["whale_se"],  capsize=5, error_kw=dict(elinewidth=1.2))
ax.bar(x + w/2, top5["retail_mean"], w, label="Retail", color="#c8d8ea",
       yerr=top5["retail_se"], capsize=5, error_kw=dict(elinewidth=1.2), edgecolor="gray")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([market_name(lbl, 30) for lbl in top5["label"]], rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Mean absolute |Δp| (2-hr window)", fontsize=10)
ax.set_title("Average Absolute Price Change After Whale vs Retail Comments\n"
             "Top 5 Markets by Volume  (± 1 SE)", fontsize=11)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "fig2_event_study_by_market.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Saved fig2")

# ──────────────────────────────────────────────────────────────────────────────
# FIG 3 — Whale vs retail |Δp| by volume quintile
# ──────────────────────────────────────────────────────────────────────────────
print("Fig 3: volume bucket analysis...")

mkt_vol = df.groupby("market_label")["market_volume"].first().reset_index()
mkt_vol["log_vol"] = np.log10(mkt_vol["market_volume"].clip(lower=1))
qs = mkt_vol["log_vol"].quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
bucket_labels = ["Q1\n<$1M", "Q2\n$1–8M", "Q3\n$8–15M", "Q4\n$15–30M", "Q5\n>$30M"]
mkt_vol["bucket"] = pd.cut(mkt_vol["log_vol"], bins=qs, labels=bucket_labels, include_lowest=True)
df_b = df.merge(mkt_vol[["market_label","bucket"]], on="market_label", how="left").dropna(subset=["bucket"])

bucket_results = []
for bucket in bucket_labels:
    sub = df_b[df_b["bucket"] == bucket]
    wh  = sub[sub["is_whale"]]["abs_price_change"].dropna()
    rt  = sub[~sub["is_whale"]]["abs_price_change"].dropna()
    if len(rt) > 0:
        bucket_results.append({"bucket": bucket,
                                "wh_mean": wh.mean() if len(wh) else np.nan,
                                "wh_se":   wh.sem()  if len(wh) else np.nan,
                                "rt_mean": rt.mean(), "rt_se": rt.sem(),
                                "n_whale": len(wh), "n_mkts": sub["market_label"].nunique()})
bdf = pd.DataFrame(bucket_results)

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(bdf)); w = 0.36
has_whale = bdf["wh_mean"].notna()
ax.bar(x[has_whale] - w/2, bdf.loc[has_whale, "wh_mean"], w,
       label="Whale", color="steelblue",
       yerr=bdf.loc[has_whale, "wh_se"], capsize=5, error_kw=dict(elinewidth=1.2))
ax.bar(x + w/2, bdf["rt_mean"], w, label="Retail", color="#c8d8ea",
       yerr=bdf["rt_se"], capsize=5, error_kw=dict(elinewidth=1.2), edgecolor="gray")
ax.axhline(0, color="black", lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels(bdf["bucket"].tolist(), fontsize=10)
ax.set_xlabel("Market Volume Quintile", fontsize=10)
ax.set_ylabel("Mean absolute |Δp| (2-hr window)", fontsize=10)
ax.set_title("Absolute Price Impact by Market Volume Quintile\nWhale vs Retail Comments  (± 1 SE)", fontsize=11)

# Annotate n_whale just above the taller bar in each group (avoids legend area)
for i, row in bdf.iterrows():
    nw = int(row.n_whale) if not np.isnan(row.n_whale) else 0
    wh_h = row.wh_mean if not np.isnan(row.wh_mean) else 0
    rt_h = row.rt_mean
    bar_top = max(wh_h, rt_h)
    se_top  = (row.wh_se if not np.isnan(row.wh_se) else 0) if wh_h >= rt_h else row.rt_se
    ax.text(i, bar_top + se_top + 0.005, f"n_whale={nw}",
            ha="center", va="bottom", fontsize=7.5, color="#444")

ax.legend(fontsize=10, loc="lower right")

plt.tight_layout()
plt.savefig(OUT / "fig3_sentiment_analysis.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Saved fig3")

# ──────────────────────────────────────────────────────────────────────────────
# FIG 4 — 3×2 confusion heatmap: Sentiment × Holder Direction (whale only)
# ──────────────────────────────────────────────────────────────────────────────
print("Fig 4: sentiment × holder direction heatmap...")

whale_df = df[df["is_whale"]].copy()

# Build (market_id_int, proxy_wallet) → 'YES'/'NO' from holders.json
holder_dir: dict[tuple[int, str], str] = {}
for p in RAW.iterdir():
    hf = p / "holders.json"
    if not hf.exists():
        continue
    try:
        mkt_id = int(p.name)
    except ValueError:
        continue
    with open(hf) as f:
        raw_h = json.load(f)
    # structure: list of {token, holders: [{proxyWallet, outcomeIndex, ...}]}
    token_groups = raw_h if isinstance(raw_h, list) else [raw_h]
    for group in token_groups:
        holders_list = group.get("holders", []) if isinstance(group, dict) else []
        # also handle flat list of holder dicts
        if not holders_list and isinstance(group, dict) and "proxyWallet" in group:
            holders_list = [group]
        for holder in holders_list:
            wallet = holder.get("proxyWallet", "")
            oi = holder.get("outcomeIndex", -1)
            if wallet and oi in (0, 1):
                holder_dir[(mkt_id, wallet)] = "YES" if oi == 0 else "NO"

whale_df["holder_direction"] = whale_df.apply(
    lambda r: holder_dir.get((int(r["market_id"]), r.get("proxy_wallet", "")), None), axis=1
)

# Restrict to events with known holder direction and valid sentiment label
known = whale_df[
    whale_df["holder_direction"].notna() &
    whale_df["sentiment_label"].notna()
].copy()
known["sentiment_label"] = known["sentiment_label"].str.lower()
n_known    = len(known)
n_excluded = len(whale_df) - n_known
n_neutral  = int((known["sentiment_label"] == "neutral").sum())

# Cross-tabulation: rows = direction (Y), cols = sentiment (X)
sent_order = ["positive", "neutral", "negative"]
dir_order  = ["YES", "NO"]
ct = pd.crosstab(known["sentiment_label"], known["holder_direction"])
ct = ct.reindex(index=sent_order, columns=dir_order, fill_value=0)
ct_pct = ct / n_known * 100

# Transpose so rows=direction, cols=sentiment → X=sentiment, Y=direction
ct_T     = ct.T.reindex(index=dir_order, columns=sent_order)
ct_pct_T = ct_pct.T.reindex(index=dir_order, columns=sent_order)

fig, ax = plt.subplots(figsize=(6, 4.5))
cmap = plt.cm.Blues
im = ax.imshow(ct_pct_T.values, cmap=cmap, vmin=0, vmax=ct_pct_T.values.max() * 1.15)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Positive", "Neutral", "Negative"], fontsize=12)
ax.set_yticks([0, 1])
ax.set_yticklabels(["YES holder", "NO holder"], fontsize=12)
ax.set_xlabel("VADER Sentiment", fontsize=11)
ax.set_ylabel("Holder Direction", fontsize=11)

for i, direction in enumerate(dir_order):
    for j, sent in enumerate(sent_order):
        n_cell = ct_T.loc[direction, sent]
        pct    = ct_pct_T.loc[direction, sent]
        color  = "white" if pct > ct_pct_T.values.max() * 0.55 else "black"
        ax.text(j, i, f"{pct:.1f}%\n(n={n_cell:,})",
                ha="center", va="center", fontsize=10, color=color, fontweight="bold")

ax.set_title(
    f"Whale Sentiment vs Holding Direction (n={n_known:,})\n"
    f"({n_neutral:,} neutral-sentiment events excluded from position-consistency test)",
    fontsize=9
)

# Consistent: YES+Positive (i=0,j=0), NO+Negative (i=1,j=2)
# Inconsistent: YES+Negative (i=0,j=2), NO+Positive (i=1,j=0)
for i, j, label in [(0, 0, "consistent"), (1, 2, "consistent"),
                     (0, 2, "inconsistent"), (1, 0, "inconsistent")]:
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                fill=False, edgecolor="orange" if "in" in label else "green",
                                lw=2.5, clip_on=False))

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="none", edgecolor="green",  linewidth=2.5, label="Consistent (talks book)"),
    Patch(facecolor="none", edgecolor="orange", linewidth=2.5, label="Inconsistent"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)
plt.tight_layout()
plt.savefig(OUT / "fig4_pump_dump.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Saved fig4")

print(f"\nKey numbers:")
print(f"  Markets total     : {n_markets_total}")
print(f"  Markets (clean)   : {n_markets_clean}")
print(f"  Markets w/ whales : {n_whale_markets}")
print(f"  Whale events      : {n_whale:,}")
print(f"  Retail events     : {n_retail:,}")
wh_abs = df[df["is_whale"]]["abs_price_change"]
rt_abs = df[~df["is_whale"]]["abs_price_change"]
print(f"  Whale |Dp| median : {wh_abs.median():.4f}  mean: {wh_abs.mean():.4f}")
print(f"  Retail |Dp| median: {rt_abs.median():.4f}  mean: {rt_abs.mean():.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# FIG 5 — CDF of |Δp| for whale vs retail
# ──────────────────────────────────────────────────────────────────────────────
print("\nFig 5: CDF of |Δp|...")

fig, ax = plt.subplots(figsize=(8, 5))

for label, mask, color, ls in [
    ("Whale", df["is_whale"],  "steelblue", "-"),
    ("Retail", ~df["is_whale"], "#c8d8ea",  "--"),
]:
    vals = df.loc[mask, "abs_price_change"].dropna().sort_values()
    cdf  = np.arange(1, len(vals) + 1) / len(vals)
    ax.plot(vals, cdf, color=color, lw=2, ls=ls, label=label)

ax.set_xlim(0, 1)
ax.set_xlabel("Absolute price change |Δp|", fontsize=11)
ax.set_ylabel("Cumulative fraction of events", fontsize=11)
ax.set_title("CDF of |Δp|: Whale vs Retail Comment Events", fontsize=12)
ax.axvline(wh_abs.mean(), color="steelblue", lw=1, ls=":", alpha=0.7,
           label=f"Whale mean ({wh_abs.mean():.3f})")
ax.axvline(rt_abs.mean(), color="#c8d8ea",  lw=1, ls=":", alpha=0.9,
           label=f"Retail mean ({rt_abs.mean():.3f})")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig5_cdf.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Saved fig5")

# ──────────────────────────────────────────────────────────────────────────────
# FIG 6 — Hour-of-day comment timing: whale vs retail
# ──────────────────────────────────────────────────────────────────────────────
print("Fig 6: comment timing by hour-of-day...")

df["hour"] = pd.to_datetime(df["timestamp"], utc=True).dt.hour

whale_hours  = df[df["is_whale"]]["hour"].value_counts().sort_index()
retail_hours = df[~df["is_whale"]]["hour"].value_counts().sort_index()

# Normalize to share of events
whale_share  = whale_hours  / whale_hours.sum()
retail_share = retail_hours / retail_hours.sum()

fig, ax = plt.subplots(figsize=(10, 5))
hours = np.arange(24)
w = 0.4
ax.bar(hours - w/2, [whale_share.get(h, 0)  for h in hours],
       w, color="steelblue", label="Whale", alpha=0.9)
ax.bar(hours + w/2, [retail_share.get(h, 0) for h in hours],
       w, color="#c8d8ea",  label="Retail", alpha=0.9, edgecolor="gray")
ax.set_xticks(hours)
ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Hour of day (UTC)", fontsize=11)
ax.set_ylabel("Share of all comment events", fontsize=11)
ax.set_title("When Do Whales Comment? Hour-of-Day Distribution (UTC)", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "fig6_timing.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Saved fig6")

# ──────────────────────────────────────────────────────────────────────────────
# FIG 7 — Reaction count vs |Δp| for whale events
# ──────────────────────────────────────────────────────────────────────────────
print("Fig 7: reaction count vs |Δp|...")

whale_df2 = df[df["is_whale"] & df["abs_price_change"].notna() &
               df["reaction_count"].notna()].copy()
whale_df2["reaction_count"] = whale_df2["reaction_count"].clip(upper=50)

# Bin by reaction count
bins = [-1, 0, 2, 5, 10, 50]
labels_r = ["0", "1–2", "3–5", "6–10", "11+"]
whale_df2["reaction_bin"] = pd.cut(whale_df2["reaction_count"],
                                    bins=bins, labels=labels_r)
bin_stats = (whale_df2.groupby("reaction_bin", observed=True)["abs_price_change"]
             .agg(mean="mean", se=lambda x: x.sem(), count="count")
             .reset_index())

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(bin_stats))
ax.bar(x, bin_stats["mean"], color="steelblue", alpha=0.9,
       yerr=bin_stats["se"], capsize=5, error_kw=dict(elinewidth=1.2))
ax.set_xticks(x)
ax.set_xticklabels(bin_stats["reaction_bin"].astype(str), fontsize=11)
ax.set_xlabel("Reactions on whale comment", fontsize=11)
ax.set_ylabel("Mean |Δp| (7-day window)", fontsize=11)
ax.set_title("Do More-Engaged Whale Comments Move Prices More?\nMean |Δp| by Reaction Count (± 1 SE)", fontsize=11)
for i, row in bin_stats.iterrows():
    ax.text(i, row["mean"] + row["se"] + 0.003, f"n={int(row['count'])}",
            ha="center", va="bottom", fontsize=8, color="#444")
plt.tight_layout()
plt.savefig(OUT / "fig7_reactions.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Saved fig7")

# ──────────────────────────────────────────────────────────────────────────────
# FIG 8 — Whale vs retail |Δp| by market category
# ──────────────────────────────────────────────────────────────────────────────
print("Fig 8: market category breakdown...")

def categorize(q):
    q = str(q).lower()
    if any(k in q for k in ["presidential", "2028", "2024", "nomination", "democrat",
                              "republican", "biden", "harris", "trump", "kamala",
                              "mayor", "governor", "senate", "win the 20"]):
        return "Elections"
    if any(k in q for k in ["fifa", "world cup"]):
        return "FIFA"
    if any(k in q for k in ["iran", "ceasefire", "ukraine", "israel", "netanyahu",
                              "russia", "hezbollah", "nuclear", "war", "invasion",
                              "nato", "zelensky", "putin", "china", "taiwan",
                              "pakistan", "india", "strike", "military"]):
        return "Geopolitics"
    if any(k in q for k in ["crypto", "bitcoin", "ethereum", "btc", "eth", "token",
                              "coin", "nft", "airdrop", "fdv", "market cap",
                              "polymarket", "tiktok", "ai ", "artificial intelligence",
                              "tesla", "spacex", "elon", "fsd", "ipo"]):
        return "Crypto & Tech"
    if any(k in q for k in ["shutdown", "prime minister", "president of", "chancellor",
                              "election", "elected", "parliament", "senate", "ban ",
                              "legislation", "tariff", "sanction", "cabinet"]):
        return "Politics"
    return "Other"

# Build question lookup
q_lookup = {}
for label in df["market_label"].unique():
    q_lookup[str(label)] = market_name(label, maxlen=200)

df["category"] = df["market_label"].astype(str).map(q_lookup).apply(categorize)

cat_order = ["Elections", "Geopolitics", "Crypto & Tech", "Politics", "FIFA", "Other"]
cat_stats = []
for cat in cat_order:
    sub = df[df["category"] == cat]
    wh  = sub[sub["is_whale"]]["abs_price_change"].dropna()
    rt  = sub[~sub["is_whale"]]["abs_price_change"].dropna()
    if len(rt) < 10:
        continue
    cat_stats.append({"cat": cat,
                       "wh_mean": wh.mean() if len(wh) else np.nan,
                       "wh_se":   wh.sem()  if len(wh) else np.nan,
                       "rt_mean": rt.mean(), "rt_se": rt.sem(),
                       "n_whale": len(wh), "n_markets": sub["market_label"].nunique()})
cdf2 = pd.DataFrame(cat_stats)

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(cdf2)); w = 0.36
has_whale = cdf2["wh_mean"].notna()
ax.bar(x[has_whale] - w/2, cdf2.loc[has_whale, "wh_mean"], w,
       color="steelblue", label="Whale",
       yerr=cdf2.loc[has_whale, "wh_se"], capsize=5, error_kw=dict(elinewidth=1.2))
ax.bar(x + w/2, cdf2["rt_mean"], w, color="#c8d8ea", label="Retail",
       yerr=cdf2["rt_se"], capsize=5, error_kw=dict(elinewidth=1.2), edgecolor="gray")
ax.set_xticks(x)
ax.set_xticklabels([f"{r['cat']}\n({r['n_markets']} mkts)" for _, r in cdf2.iterrows()],
                    fontsize=10)
ax.set_ylabel("Mean |Δp| (7-day window)", fontsize=11)
ax.set_title("Whale vs Retail Price Impact by Market Category (± 1 SE)", fontsize=12)
for i, row in cdf2.iterrows():
    nw = int(row.n_whale) if not np.isnan(row.n_whale) else 0
    if nw > 0:
        top = (row.wh_mean if not np.isnan(row.wh_mean) else 0)
        se  = (row.wh_se  if not np.isnan(row.wh_se)  else 0)
        ax.text(i - w/2, top + se + 0.003, f"n={nw}",
                ha="center", va="bottom", fontsize=8, color="#444")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "fig8_categories.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Saved fig8")

# ──────────────────────────────────────────────────────────────────────────────
# FIG 9 — Scatter: position size vs price impact (whale events only)
# ──────────────────────────────────────────────────────────────────────────────
print("\nFig 9: scatter position size vs |Δp|...")

whale_scatter = df[df["is_whale"] & df["abs_price_change"].notna() &
                   (df["position_usd"] >= WHALE_THRESHOLD)].copy()
whale_scatter["log_pos"] = np.log10(whale_scatter["position_usd"].clip(lower=1))

bin_edges9 = np.percentile(whale_scatter["log_pos"], np.linspace(0, 100, 11))
bin_edges9 = np.unique(bin_edges9)
whale_scatter["pos_bin"] = pd.cut(whale_scatter["log_pos"], bins=bin_edges9, include_lowest=True)
bin_stats9 = (whale_scatter.groupby("pos_bin", observed=True)
              .agg(mean=("abs_price_change", "mean"),
                   se=("abs_price_change", lambda x: x.sem()),
                   count=("abs_price_change", "count"),
                   mid=("log_pos", "mean"))
              .reset_index())

fig, ax = plt.subplots(figsize=(9, 5.5))
sample9 = whale_scatter.sample(min(3000, len(whale_scatter)), random_state=42)
ax.scatter(sample9["log_pos"], sample9["abs_price_change"],
           alpha=0.12, s=8, color="steelblue", zorder=1)
ax.errorbar(bin_stats9["mid"], bin_stats9["mean"],
            yerr=bin_stats9["se"], fmt="o-", color="navy",
            capsize=4, lw=2, ms=7, zorder=5, label="Bin mean (± 1 SE)")
for _, row in bin_stats9.iterrows():
    ax.text(row["mid"], row["mean"] + row["se"] + 0.005,
            f"n={int(row['count'])}", ha="center", fontsize=7, color="#444")

tick_vals9 = [np.log10(v) for v in [5000, 10000, 50000, 100000, 500000, 1000000]]
tick_labs9 = ["$5K", "$10K", "$50K", "$100K", "$500K", "$1M"]
ax.set_xticks(tick_vals9)
ax.set_xticklabels(tick_labs9, fontsize=9)
ax.set_xlabel("Whale position size (USD, log scale)", fontsize=11)
ax.set_ylabel("Absolute price change |Δp|", fontsize=11)
ax.set_title(f"Does Position Size Predict Price Impact?\n"
             f"Whale Events Only (n={len(whale_scatter):,})", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(OUT / "fig9_scatter_position_impact.png", bbox_inches="tight", dpi=150)
plt.close()
print("  Saved fig9")
