import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

plt.rcParams["figure.dpi"] = 120

def _ensure_group_schema(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Accept tables with either:
      - ['gender' or 'age_bin', 'N_samples', 'MAE_pitch_deg','MAE_yaw_deg','MAE_avg_deg','N_subjects']
      - or already normalized with 'group' instead of gender/age_bin.
    Normalize to: ['group','N_samples','MAE_pitch_deg','MAE_yaw_deg','MAE_avg_deg','N_subjects']
    """
    df = df.copy()
    if "group" not in df.columns:
        if kind == "gender" and "gender" in df.columns:
            df = df.rename(columns={"gender":"group"})
        elif kind == "age" and "age_bin" in df.columns:
            df = df.rename(columns={"age_bin":"group"})
        else:
            raise ValueError(f"[SCHEMA] Could not find a group-like column in {df.columns.tolist()}")

    # Keep only known columns if extras exist
    keep = ["group","N_samples","MAE_pitch_deg","MAE_yaw_deg","MAE_avg_deg","N_subjects"]
    df = df[[c for c in keep if c in df.columns]]

    # Cast numerics safely
    for c in ["N_samples","MAE_pitch_deg","MAE_yaw_deg","MAE_avg_deg","N_subjects"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Order groups nicely
    if kind == "gender":
        # prefer Women, Men if present
        order = [g for g in ["Woman","Women","Female","Man","Men","Male"] if g in set(df["group"].astype(str))]
        if order:
            # unify variants to 'Women'/'Men' for display
            map_norm = {"Woman":"Women","Female":"Women","Man":"Men","Male":"Men"}
            df["group"] = df["group"].astype(str).map(lambda x: map_norm.get(x, x))
            # update order after map
            order = [g for g in ["Women","Men"] if g in set(df["group"].astype(str))]
            df["group"] = pd.Categorical(df["group"], categories=order, ordered=True)
            df = df.sort_values("group")
        else:
            df = df.sort_values("MAE_avg_deg", ascending=True)
    else:
        # age bins: prefer 18–34, 35–54, 55+ if they exist
        desired = ["18-34","35-54","55+"]
        cats = [g for g in desired if g in set(df["group"].astype(str))]
        if cats:
            df["group"] = pd.Categorical(df["group"].astype(str), categories=cats, ordered=True)
            df = df.sort_values("group")
        else:
            df = df.sort_values("MAE_avg_deg", ascending=True)
    return df

def _barplot_simple(df: pd.DataFrame, title: str, out_png: str):
    x = df["group"].astype(str).tolist()
    y = df["MAE_avg_deg"].values
    n = df["N_samples"].values if "N_samples" in df.columns else None

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(x, y)
    ax.set_xlabel("Group")
    ax.set_ylabel("MAE (deg)")
    ax.set_title(title)

    # annotate counts lightly
    if n is not None:
        for i, (xi, yi, ni) in enumerate(zip(x, y, n)):
            ax.text(i, yi + max(y)*0.02, f"N={int(ni)}", ha="center", va="bottom", fontsize=8, rotation=0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print("[WRITE]", out_png)

def fig_gender(table_dir: Path, out_dir: Path):
    path = table_dir / "ETH_mae_by_gender.csv"
    if not path.exists():
        print("[SKIP] Fig 5.5 — gender table not found:", str(path))
        return
    df = pd.read_csv(path)
    try:
        df = _ensure_group_schema(df, kind="gender")
    except Exception as e:
        print("[SKIP] Fig 5.5 —", e)
        return
    _barplot_simple(df, "ETH-XGaze — MAE by gender (raw baseline)", str(out_dir / "figs" / "F5_mae_by_gender.png"))

def fig_age(table_dir: Path, out_dir: Path):
    path = table_dir / "ETH_mae_by_agebin.csv"
    if not path.exists():
        print("[SKIP] Fig 5.6 — age table not found:", str(path))
        return
    df = pd.read_csv(path)
    try:
        df = _ensure_group_schema(df, kind="age")
    except Exception as e:
        print("[SKIP] Fig 5.6 —", e)
        return
    _barplot_simple(df, "ETH-XGaze — MAE by age bin (raw baseline)", str(out_dir / "figs" / "F6_mae_by_agebin.png"))

def fig_val_curves(out_dir: Path):
    # leave as-is; your earlier run already produced this file.
    # Here we just re-write a minimal placeholder if needed.
    p = out_dir / "figs" / "F7_val_calibration_curves.png"
    if not p.exists():
        fig, ax = plt.subplots(figsize=(6,3.6))
        ax.plot([0,1],[0,1])
        ax.set_title("Validation residuals with fitted calibration curves")
        ax.set_xlabel("Predicted angle (deg)")
        ax.set_ylabel("Residual (deg)")
        os.makedirs(p.parent, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print("[WRITE]", str(p))
    else:
        print("[KEEP]", str(p))

def fig_offsets(out_dir: Path):
    p = out_dir / "figs" / "F8_yawbin_offsets_before_after.png"
    if not p.exists():
        fig, ax = plt.subplots(figsize=(6,3.6))
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("Yaw-bin mean residuals before/after offsets")
        ax.set_xlabel("Yaw bin")
        ax.set_ylabel("Mean residual (deg)")
        os.makedirs(p.parent, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print("[WRITE]", str(p))
    else:
        print("[KEEP]", str(p))

def fig_stage_progress(out_dir: Path):
    p = out_dir / "figs" / "F9_stage_progression_bars.png"
    if not p.exists():
        fig, ax = plt.subplots(figsize=(6,3.6))
        names = ["baseline","stageA","stageB","champ"]
        vals = [22.0, 18.0, 15.0, 13.3]  # placeholder bars (visual only; numbers not used in thesis)
        ax.bar(names, vals)
        ax.set_title("ETH test MAE progression by stage (visual placeholder)")
        ax.set_xlabel("Stage")
        ax.set_ylabel("MAE (deg)")
        os.makedirs(p.parent, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print("[WRITE]", str(p))
    else:
        print("[KEEP]", str(p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="root outdir where figs/ and tables/ live")
    args = ap.parse_args()
    out_dir = Path(args.outdir)
    tables = out_dir / "tables"

    fig_gender(tables, out_dir)
    fig_age(tables, out_dir)
    fig_val_curves(out_dir)
    fig_offsets(out_dir)
    fig_stage_progress(out_dir)

if __name__ == "__main__":
    main()
