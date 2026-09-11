"""Build Appendix B MAE table from Stage 41 summary (T-GCN family, Los + SZ)."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "gsl_stage41" / "stage41_summary.csv"
OUT = Path(__file__).resolve().parent / "sections" / "appendix_mae.tex"

METHODS = [
    ("physical", "T-GCN"),
    ("no_spatial", "T-GCN-NoSpatial"),
    ("gsl", "T-GCN-GSL"),
    ("cgsl", "T-GCN-cGSL"),
    ("multi_gsl", "T-GCN-MultiGSL"),
    ("multi_gsl_weighted", "T-GCN-MultiGSL-Weighted"),
    ("multi_gsl_mix", "T-GCN-MultiGSL-Mix"),
    ("gcn_physical", "GCN"),
    ("gcn_no_spatial", "GCN-NoSpatial"),
    ("gcn_gsl", "GCN-GSL"),
    ("gcn_cgsl", "GCN-cGSL"),
    ("gcn_multigsl", "GCN-MultiGSL"),
]

rows = {}
with open(SRC, newline="") as f:
    for r in csv.DictReader(f):
        key = (r["dataset"], int(r["ph"]), r["variant_id"])
        rows[key] = (float(r["mae_mean"]), float(r["mae_std"]))


def fmt(dataset: str, variant: str, ph: int) -> str:
    m, s = rows[(dataset, ph, variant)]
    return f"{m:.2f} ({s:.2f})"


lines = [
    "% Auto-generated from gsl_stage41/stage41_summary.csv",
    "\\section{Additional MAE Results}\\label{sec:app-mae}",
    "",
    "Table~\\ref{tab:mae} reports mean MAE (sample standard deviation in",
    "parentheses) over five seeds on the de-normalized scale, parallel to the",
    "main RMSE tables.",
    "",
    "\\begin{table}[t]",
    "\\centering",
    "\\caption{Test MAE (mean, sample standard deviation) over five seeds.}",
    "\\label{tab:mae}",
    "\\small",
    "\\begin{tabular}{lcccc}",
    "\\toprule",
    "Method & PH1 & PH2 & PH3 & PH4 \\\\",
    "\\midrule",
]

for ds_name, ds_key in (("Los-loop", "losloop"), ("SZ-Taxi", "shenzhen")):
    lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{ds_name}}}}} \\\\")
    for vid, name in METHODS:
        cells = " & ".join(fmt(ds_key, vid, ph) for ph in range(1, 5))
        lines.append(f"{name} & {cells} \\\\")
    lines.append("\\midrule")

lines[-1] = "\\bottomrule"
lines += ["\\end{tabular}", "\\end{table}", ""]
OUT.write_text("\n".join(lines), encoding="utf-8")
print("Wrote", OUT)
