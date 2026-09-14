from pathlib import Path
import shutil

ROOT = Path("C:/git/mamintoosi/TGCN-GSL-PyTorch")
ARCH = ROOT / "archive" / "repo_cleanup_20260913"
ARCH.mkdir(parents=True, exist_ok=True)

# Top-level stage briefs and obsolete runners
root_moves = [
    "Stage42.md",
    "Stage43.md",
    "Stage44.md",
    "Stage45.md",
    "Stage45_1.md",
    "Stage46.md",
    "Stage47.md",
    "Stage48.md",
    "run_regenerate_figures.sh",
    "run_resolution_experiment.sh",
    "run_stage29_los15min.sh",
    "run_stage32_sparse_control.sh",
    "run_stage33A_sparse_control.sh",
    "run_stage33B_gsl_canonical.sh",
    "run_stage33C_sz_multiseed.sh",
    "run_stage40_3_sz_dagma.sh",
    "run_stage40_3_validate.py",
    "changes.bundle",
]

# Directory trees to archive (keep live experiment cores out of this list)
dir_moves = [
    "gsl_stage44",
    "gsl_stage45",
    "gsl_stage45_1",
    "gsl_stage46",
    "gsl_stage47",
    "gsl_stage48",
    "gsl_stage49",
    "gsl_stage50",
    "gsl_stage51",
    "gsl_master",
    "gsl_stage42",  # claim audit docs only for writing; stage41 stays
    "gsl_stage57_tgcn_gcn_audit",  # report + audit scripts; results already in results/
]

# paper subfolders: submitted/previous are historical; keep revised_version only live
paper_moves = [
    "paper/previous_revision",
    "paper/submitted_version",
]

# old run scripts at paper/revised_version if any duplicates — leave scripts/

# results: keep stage40, stage26_validation (DAGMA npy), stage26_checkpoint, stage29, stage32, stage58
# archive forensic / manuscript recon only
results_moves = [
    "results/stage27_resolution",
    "results/stage30_forensic_audit",
    "results/stage31_manuscript_integration",
    "results/stage31_manuscript_reconstruction",
    "results/stage57_tgcn_gcn_audit",
]

# doc: archive entire stage-report corpus; keep empty or move all
doc_all = True

moved = []
for rel in root_moves + dir_moves + paper_moves + results_moves:
    src = ROOT / rel
    if not src.exists():
        print("skip missing", rel)
        continue
    dst = ARCH / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print("already archived", rel)
        continue
    shutil.move(str(src), str(dst))
    moved.append(rel)
    print("archived", rel)

# Move doc/ wholesale
doc = ROOT / "doc"
if doc.exists():
    dst = ARCH / "doc"
    if not dst.exists():
        shutil.move(str(doc), str(dst))
        moved.append("doc")
        print("archived doc/")
    else:
        print("doc already archived")

# Keep gsl_stage41 scripts but it's writing-heavy — user asked unused reports.
# stage41 is needed for summary CSV? summary already exists under gsl_stage41.
# KEEP gsl_stage41 (audit) and gsl_stage40, gsl_stage26, gsl_stage58.

print("\nMoved", len(moved), "items")
print("Archive at", ARCH)
print("Remaining top-level:")
for p in sorted(ROOT.iterdir()):
    print(" ", p.name)
