"""
Regression analysis: do whale comments predict price changes on Polymarket?

Runs four OLS models with HC3 heteroskedasticity-robust standard errors
on the pooled cross-market event dataset.

Models
------
M1: price_change ~ whale_int + C(market_label)
    Baseline: whale-comment dummy, market fixed effects.

M2: price_change ~ whale_int + sentiment_compound + C(market_label)
    Add VADER sentiment as a control.

M3: price_change ~ whale_int * log_market_volume + sentiment_compound
    Interaction: does whale impact scale with market size?

M4 (holders subsample):
    price_change ~ whale_int * log_market_volume + log_position + sentiment_compound
    Add log(position_usd) to test whether larger positions drive larger moves.

Usage
-----
    python code/04_analyze.py
    python code/04_analyze.py --input data/processed/pooled_events.csv
    python code/04_analyze.py --output-dir output/

Outputs
-------
    output/regression_table.tex   — booktabs LaTeX table (4 columns)
    output/summary_stats.csv      — summary statistics for key variables
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DEFAULT_INPUT  = ROOT / "data" / "processed" / "pooled_events.csv"
DEFAULT_OUTPUT = ROOT / "output"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """Load pooled_events.csv and add analysis columns."""
    print("Loading pooled_events.csv...")
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    n_rows    = len(df)
    n_markets = df["market_label"].nunique() if "market_label" in df.columns else "?"
    print(f"Loaded {n_rows:,} rows across {n_markets} markets")
    return df


def apply_convergence_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove observations where price was already converging to 0/1
    (price_before < 0.10 or > 0.90) — those price changes are noise.
    Also require a non-null price_change.
    """
    print("\nApplying convergence filter...")
    before = len(df)
    df = df[df["price_change"].notna()].copy()
    df = df[~df["is_convergence_period"]].copy()
    after = len(df)

    n_whale  = int(df["is_whale"].sum())
    n_retail = after - n_whale
    print(f"{after:,} non-convergence observations, {n_whale:,} whale events "
          f"({n_retail:,} retail events)")
    print(f"  (dropped {before - after:,} rows: convergence period or missing price)")
    return df


def add_regression_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Add whale_int (0/1 integer) from is_whale bool."""
    df = df.copy()
    df["whale_int"] = df["is_whale"].astype(int)
    return df


def run_regressions(df: pd.DataFrame) -> dict:
    """
    Fit M1–M4 with HC3 standard errors.
    Returns dict of {name: (result, formula, n_obs)} for each model.
    """
    print("\nRunning OLS regressions...")

    # M1 baseline — whale dummy + market FE
    f1 = "price_change ~ whale_int + C(market_label)"
    r1 = smf.ols(f1, data=df).fit(cov_type="HC3")
    print_model_summary("M1", r1, f1)

    # M2 — add sentiment
    f2 = "price_change ~ whale_int + sentiment_compound + C(market_label)"
    r2 = smf.ols(f2, data=df).fit(cov_type="HC3")
    print_model_summary("M2", r2, f2)

    # M3 — interaction with log market volume
    f3 = "price_change ~ whale_int * log_market_volume + sentiment_compound"
    df3 = df[df["log_market_volume"].notna()].copy()
    r3 = smf.ols(f3, data=df3).fit(cov_type="HC3")
    print_model_summary("M3", r3, f3)

    # M4 — holders subsample: add log_position
    df4 = df3[df3["log_position"].notna() & (df3["position_usd"] > 0)].copy()
    f4 = "price_change ~ whale_int * log_market_volume + log_position + sentiment_compound"
    r4 = smf.ols(f4, data=df4).fit(cov_type="HC3")
    print_model_summary("M4 (holders)", r4, f4)

    return {
        "M1": (r1, f1, int(r1.nobs)),
        "M2": (r2, f2, int(r2.nobs)),
        "M3": (r3, f3, int(r3.nobs)),
        "M4": (r4, f4, int(r4.nobs)),
    }


def print_model_summary(label: str, result, formula: str) -> None:
    """Print key coefficients for one model."""
    print(f"\n  {label}: {formula}")
    print(f"  N={int(result.nobs):,}  R2={result.rsquared:.4f}  adj-R2={result.rsquared_adj:.4f}")
    FOCUS = ["whale_int", "sentiment_compound", "log_market_volume",
             "whale_int:log_market_volume", "log_position"]
    for param in FOCUS:
        if param in result.params:
            coef  = result.params[param]
            se    = result.bse[param]
            pval  = result.pvalues[param]
            stars = _stars(pval)
            print(f"    {param:<35s}  coef={coef:+.5f}  se={se:.5f}  p={pval:.4f}{stars}")


def _stars(p: float) -> str:
    if p < 0.01:  return "  ***"
    if p < 0.05:  return "  **"
    if p < 0.10:  return "  *"
    return ""


def _fmt_coef(result, param: str, digits: int = 4) -> str:
    """Return 'coef^stars (se)' string or empty string if param not in model."""
    if param not in result.params:
        return ""
    coef  = result.params[param]
    se    = result.bse[param]
    pval  = result.pvalues[param]
    stars = _stars(pval).strip()
    return f"${coef:+.{digits}f}^{{{stars}}}$ ({se:.{digits}f})"


def build_latex_table(models: dict) -> str:
    """
    Build a four-column booktabs regression table.

    Rows: whale_int, sentiment_compound, log_market_volume,
          whale_int:log_market_volume, log_position, N, R2.
    """
    r1, _, n1 = models["M1"]
    r2, _, n2 = models["M2"]
    r3, _, n3 = models["M3"]
    r4, _, n4 = models["M4"]
    results = [r1, r2, r3, r4]
    ns      = [n1, n2, n3, n4]

    PARAMS = [
        ("whale_int",                    "Whale comment (0/1)"),
        ("sentiment_compound",           "Sentiment compound"),
        ("log_market_volume",            "Log market volume"),
        ("whale_int:log_market_volume",  "Whale $\\times$ log volume"),
        ("log_position",                 "Log position size"),
    ]

    def row(label: str, param: str) -> str:
        cells = [_fmt_coef(r, param) for r in results]
        # pad empty cells so LaTeX sees 4 columns
        cells = [c if c else "" for c in cells]
        return f"    {label} & " + " & ".join(cells) + " \\\\\n"

    lines = []
    lines.append(r"\begin{table}[htbp]" + "\n")
    lines.append(r"  \centering" + "\n")
    lines.append(r"  \caption{OLS Regression Results: Whale Comments and Price Changes}" + "\n")
    lines.append(r"  \label{tab:regression}" + "\n")
    lines.append(r"  \begin{tabular}{lcccc}" + "\n")
    lines.append(r"    \toprule" + "\n")
    lines.append(r"    & M1 & M2 & M3 & M4 (Holders) \\" + "\n")
    lines.append(r"    \midrule" + "\n")

    for param, label in PARAMS:
        lines.append(row(label, param))

    lines.append(r"    \midrule" + "\n")
    # N row
    n_cells = " & ".join(f"{n:,}" for n in ns)
    lines.append(f"    $N$ & {n_cells} \\\\\n")
    # R2 row
    r2_cells = " & ".join(f"{r.rsquared:.3f}" for r in results)
    lines.append(f"    $R^2$ & {r2_cells} \\\\\n")

    lines.append(r"    \bottomrule" + "\n")
    lines.append(r"  \end{tabular}" + "\n")
    lines.append(
        "  \\begin{tablenotes}\\small\n"
        "    \\item HC3 heteroskedasticity-robust standard errors in parentheses.\n"
        "    \\item $^{*}p<0.10$,\\ $^{**}p<0.05$,\\ $^{***}p<0.01$.\n"
        "    \\item M4 restricts to observations with a known position size.\n"
        "  \\end{tablenotes}\n"
    )
    lines.append(r"\end{table}" + "\n")

    return "".join(lines)


def save_summary_stats(df: pd.DataFrame, out_dir: Path) -> None:
    """Save summary statistics for key variables to output/summary_stats.csv."""
    cols = [
        "price_change", "whale_int", "sentiment_compound",
        "log_market_volume", "position_usd", "log_position",
    ]
    available = [c for c in cols if c in df.columns]
    stats = df[available].describe(percentiles=[0.25, 0.5, 0.75]).T
    out = out_dir / "summary_stats.csv"
    stats.to_csv(out)
    print(f"\nSummary stats saved to {out}")


def print_position_direction_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the position-direction analysis:
    - How often do whale comments align with the commenter's financial position?
    - How often does the post-comment price move benefit the whale holder?
    - Pump/dump signal counts.
    """
    whale = df[df["is_whale"]]
    print("\n=== POSITION-DIRECTION ANALYSIS ===")

    # Sentiment consistency
    if "sentiment_consistent_with_position" in whale.columns:
        scp = whale["sentiment_consistent_with_position"].dropna()
        if len(scp):
            n_con = int(scp.sum())
            n_ctr = int((scp == 0).sum())
            print(f"Sentiment consistent with position: {n_con}/{len(scp)} ({100*n_con/len(scp):.1f}%)")
            print(f"Sentiment contrarian to position:  {n_ctr}/{len(scp)} ({100*n_ctr/len(scp):.1f}%)")
            print(f"  (50% = random; >50% = talks own book; <50% = contrarian bias)")
    else:
        print("  [sentiment_consistent_with_position column not found — re-run process.py]")

    # Price direction benefit
    if "price_helped_holder" in whale.columns:
        php = whale["price_helped_holder"].dropna()
        if len(php):
            n_h = int(php.sum())
            print(f"Post-comment price helped holder:  {n_h}/{len(php)} ({100*n_h/len(php):.1f}%)")
            print(f"  (50% = random; >50% = price moves favour whale)")
    else:
        print("  [price_helped_holder column not found — re-run process.py]")

    # Pump / dump
    for col, label in [("pump_signal", "Pump"), ("dump_signal", "Dump")]:
        if col in whale.columns:
            n = int(whale[col].fillna(False).sum())
            print(f"{label} signals: {n}")
        else:
            print(f"  [{col} column not found]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run whale-comment regression analysis on Polymarket data"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Path to pooled_events.csv (default: data/processed/pooled_events.csv)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help="Directory for output files (default: output/)"
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    df = load_data(args.input)

    # 2. Filter
    df = apply_convergence_filter(df)

    # 3. Prep regression variables
    df = add_regression_vars(df)

    # 4. Run regressions
    models = run_regressions(df)

    # 5. Save LaTeX table
    print("\nBuilding LaTeX regression table...")
    latex = build_latex_table(models)
    tex_path = out_dir / "regression_table.tex"
    tex_path.write_text(latex)
    print(f"Regression table saved to {tex_path}")

    # 6. Save summary stats
    save_summary_stats(df, out_dir)

    # 7. Position-direction summary
    print_position_direction_summary(df)

    print("\nDone. All outputs written to", out_dir)


if __name__ == "__main__":
    main()
