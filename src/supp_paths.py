"""
Isolated output-path helpers for the supplementary PH5/PH6 experiments
(rev2 per todo5: 10-variant matrix, two separate roots, guarded copying of
canonical artifacts with full provenance).

TWO SUPPLEMENTARY ROOTS (rev2)
------------------------------
  canonical-completion experiments (lambda = per-dataset canonical):
      results/supplementary_ph56/
  lambda_1 = 0.01 sensitivity experiment (Los-loop only):
      results/supplementary_ph56_lambda001/

PROTECTION GUARANTEES
---------------------
* stdlib only at module level (numpy imported lazily inside save helpers);
* every write path is derived from the caller's output root; the root is
  safety-checked against the canonical result trees AND against the other
  supplementary root (mutual isolation, todo5 #3 and #8);
* every artifact filename encodes dataset, PH, lambda tag, graph type /
  variant, and seed (where applicable) so lambda_1 = 0.01 and 0.02 can
  never collide;
* save/copy helpers REFUSE to overwrite existing files unless
  overwrite=True is passed explicitly; default is no overwrite;
* atomic_copy_with_provenance() copies canonical artifacts READ-ONLY with
  respect to the source and writes a sidecar provenance JSON (source path,
  sha256, dataset, PH, lambda, graph type, copy timestamp, and an explicit
  statement that the source was not modified) — no silent copying.

Lambda tag convention: lambda_tag(0.01) == "lambda001", lambda_tag(0.02)
== "lambda002" (matches the todo4/todo5 example filenames).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------------
# Conventions (mirrors of the canonical pipeline's, kept in one place)
# ----------------------------------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]                    # canonical seed convention
DATASETS = ["losloop", "shenzhen"]
DATASET_PREFIX = {"losloop": "los", "shenzhen": "sz"}
LAMBDA_CANONICAL = {"losloop": 0.02, "shenzhen": 0.01}

# 10-variant supplementary matrix (todo5 #1: MultiGSL-Weighted REMOVED).
# "GraphFree" is the paper's name for the no_spatial identity control.
SUPP_VARIANTS = [
    # T-GCN family
    "no_spatial", "physical", "gsl", "cgsl", "multi_gsl",
    # GCN family
    "gcn_no_spatial", "gcn_physical", "gcn_gsl", "gcn_cgsl", "gcn_multigsl",
]
EXPERIMENT_KINDS = ["canonical_completion", "lambda_sensitivity"]

DEFAULT_COMPLETION_ROOT = Path("results") / "supplementary_ph56"
DEFAULT_SENSITIVITY_ROOT = Path("results") / "supplementary_ph56_lambda001"

# Never a write target: the four canonical result trees, results/ itself, and
# EITHER supplementary root (mutual isolation between the two experiment kinds).
PROTECTED_CANONICAL_DIRS = [
    "results/stage26_validation",
    "results/stage33_gsl_canonical",
    "results/stage40_canonical",
    "results/stage59_ph56_horizon",
]
PROTECTED_SUPP_DIRS = [str(DEFAULT_COMPLETION_ROOT), str(DEFAULT_SENSITIVITY_ROOT)]
# Always protected: canonical trees + results/ itself. The two supplementary
# roots protect EACH OTHER (a canonical_completion invocation must not write
# into the sensitivity root and vice versa) — handled kind-aware in
# check_root_safety below, so a kind's own default root stays writable.
PROTECTED_EXACT_DIRS = PROTECTED_CANONICAL_DIRS + ["results"]


def lambda_tag(lambda1: float) -> str:
    """0.01 -> 'lambda001', 0.02 -> 'lambda002'."""
    return f"lambda{int(round(lambda1 * 100)):03d}"


def default_root_for_kind(kind: str) -> Path:
    if kind == "canonical_completion":
        return DEFAULT_COMPLETION_ROOT
    if kind == "lambda_sensitivity":
        return DEFAULT_SENSITIVITY_ROOT
    raise ValueError(f"Unknown experiment kind: {kind}")


# ----------------------------------------------------------------------------
# Overwrite / root protection
# ----------------------------------------------------------------------------
def check_root_safety(output_root, kind: str | None = None) -> Path:
    """Refuse output roots that could endanger canonical or other-kind results.

    With kind given, the OTHER supplementary root is additionally refused even
    if the caller renamed directories (exact-match list covers the defaults;
    renamed roots cannot be detected, so defaults + canonical trees are the
    enforced guarantee).
    """
    root = Path(output_root)
    try:
        rp = root.resolve()
    except OSError as exc:  # pragma: no cover
        raise ValueError(f"Cannot resolve --output-root {output_root}: {exc}")

    # (a) Equality refusals: canonical trees, results/ itself, and the OTHER
    #     kind's default root (mutual isolation between experiment kinds).
    other_kind_root = None
    if kind == "canonical_completion":
        other_kind_root = DEFAULT_SENSITIVITY_ROOT
    elif kind == "lambda_sensitivity":
        other_kind_root = DEFAULT_COMPLETION_ROOT
    exact_refusals = list(PROTECTED_CANONICAL_DIRS) + ["results"]
    if other_kind_root is not None:
        exact_refusals.append(str(other_kind_root))
    else:
        # kind unknown: protect both supplementary defaults conservatively.
        exact_refusals.extend(PROTECTED_SUPP_DIRS)

    for rel in exact_refusals:
        if rp == Path(rel).resolve():
            raise ValueError(
                f"--output-root {output_root} is a protected directory "
                f"('{rel}'). Use the dedicated root for this experiment kind "
                "(see default_root_for_kind) or a clearly separate "
                "directory.")

    # (b) Containment refusals: never INSIDE a canonical stage tree, and
    #     never inside the other kind's root.
    inside_refusals = list(PROTECTED_CANONICAL_DIRS)
    if other_kind_root is not None:
        inside_refusals.append(str(other_kind_root))
    for rel in inside_refusals:
        fp = Path(rel).resolve()
        inside = False
        try:
            inside = rp.is_relative_to(fp)  # Python >= 3.9
        except AttributeError:  # pragma: no cover
            inside = str(rp).startswith(str(fp) + os.sep)
        if inside:
            raise ValueError(
                f"--output-root {output_root} lies inside the protected "
                f"directory '{rel}'. Choose a separate directory.")
    return root


def guard_output(path, overwrite: bool = False) -> None:
    """Refuse to clobber an existing file unless overwrite is explicit."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}\n"
            "(default is no-overwrite; pass --overwrite explicitly if this "
            "is really intended)")


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Guarded, atomic save / copy helpers
# ----------------------------------------------------------------------------
def atomic_save_npy(path, array, overwrite: bool = False) -> Path:
    """Guarded atomic .npy write (numpy imported lazily)."""
    import numpy as np  # lazy: keeps this module importable without numpy

    path = Path(path)
    guard_output(path, overwrite)
    ensure_dir(path.parent)
    tmp = path.with_name(path.stem + ".tmp.npy")
    try:
        np.save(tmp, array)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def atomic_save_json(path, payload, overwrite: bool = False) -> Path:
    """Guarded atomic .json write."""
    path = Path(path)
    guard_output(path, overwrite)
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def atomic_copy_with_provenance(src, dst, meta_extra: dict | None = None,
                                overwrite: bool = False) -> Path:
    """Guarded copy of a canonical artifact into the supplementary root.

    The source is opened READ-ONLY (shutil.copy2 reads only; it never
    touches the source). Writes: (1) the copy at dst (refused if it exists
    unless overwrite=True), (2) a provenance sidecar dst.with_suffix(
    '.provenance.json') recording source path, sha256, byte size, copy
    timestamp, any extra metadata (dataset/PH/lambda/graph type), and an
    explicit statement that the source artifact was not modified.
    """
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source artifact missing: {src}")
    guard_output(dst, overwrite)
    ensure_dir(dst.parent)
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        shutil.copy2(src, tmp)          # read-only w.r.t. src
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()
    prov = {
        "provenance_kind": "copied_from_canonical",
        "source_path": str(src.resolve()),
        "source_sha256": sha256_of(src),
        "source_bytes": int(Path(src).stat().st_size),
        "copied_to": str(dst),
        "copy_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_not_modified": ("the source artifact was opened read-only "
                                "and was not modified by this copy"),
    }
    if meta_extra:
        prov.update(meta_extra)
    atomic_save_json(dst.with_name(dst.stem + ".provenance.json"), prov,
                     overwrite=overwrite)
    return dst


# ----------------------------------------------------------------------------
# Path builders (names encode dataset, PH, lambda, graph type, seed)
# ----------------------------------------------------------------------------
def graph_dir(output_root, lambda1: float) -> Path:
    """One graph subdirectory per lambda value: 0.02 and 0.01 never mix."""
    return Path(output_root) / f"graphs_{lambda_tag(lambda1)}"


def multilag_blocks_dir(output_root, lambda1: float) -> Path:
    return graph_dir(output_root, lambda1) / "multilag_blocks"


def graph_paths(output_root, dataset: str, ph: int, lambda1: float, gtype: str):
    """gtype: 'gsl' or 'cgsl' (cgsl is symmetrized from the gsl W_est)."""
    d = graph_dir(output_root, lambda1)
    tag = lambda_tag(lambda1)
    base = d / f"{DATASET_PREFIX[dataset]}_ph{ph}_{tag}_{gtype}"
    return {
        "A": base.with_name(base.name + "_A_binary.npy"),
        "W": base.with_name(base.name + "_W_est.npy"),
        "meta": base.with_name(base.name + "_meta.json"),
    }


def multilag_block_copy_path(output_root, dataset: str, lambda1: float, lag: int):
    """Copied Stage 26 block (PH-independent), renamed without PH ambiguity."""
    d = multilag_blocks_dir(output_root, lambda1)
    prefix = DATASET_PREFIX[dataset]
    tag = lambda_tag(lambda1)
    return d / f"{prefix}_multilag_lag{lag}_from_ph1_seed42_{tag}.npy"


def multilag_provenance_path(output_root, dataset: str, lambda1: float):
    d = multilag_blocks_dir(output_root, lambda1)
    prefix = DATASET_PREFIX[dataset]
    tag = lambda_tag(lambda1)
    return d / f"{prefix}_multilag_blocks_provenance_{tag}.json"


def training_dir(output_root, lambda1: float) -> Path:
    return Path(output_root) / f"training_{lambda_tag(lambda1)}"


def training_path(output_root, dataset: str, ph: int, seed: int, variant: str,
                  lambda1: float) -> Path:
    return (training_dir(output_root, lambda1)
            / f"{dataset}_ph{ph}_seed{seed}_{variant}_{lambda_tag(lambda1)}.json")


def summary_paths(output_root):
    root = Path(output_root)
    return {"json": root / "supplementary_summary.json",
            "csv": root / "supplementary_summary.csv"}


def log_dir(output_root) -> Path:
    return Path(output_root) / "logs"


# ----------------------------------------------------------------------------
# Shared CLI plumbing
# ----------------------------------------------------------------------------
def add_root_args(parser):
    parser.add_argument("--experiment-kind", type=str,
                        default="canonical_completion",
                        choices=EXPERIMENT_KINDS,
                        help="which supplementary experiment this invocation "
                             "belongs to; determines the default output root "
                             "(canonical_completion -> "
                             "results/supplementary_ph56, lambda_sensitivity "
                             "-> results/supplementary_ph56_lambda001) and "
                             "the lambda policy")
    parser.add_argument("--output-root", type=str, default=None,
                        help="override the kind-based output root (defaults: "
                             "see --experiment-kind). Never point this at a "
                             "canonical results directory or the other "
                             "kind's root — it will be refused.")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow overwriting THIS kind's own supplementary "
                             "outputs (default: refuse). Canonical results "
                             "are protected regardless of this flag.")
    return parser


def resolve_root(args) -> Path:
    kind = args.experiment_kind
    root_arg = getattr(args, "output_root", None)
    root = Path(root_arg) if root_arg else default_root_for_kind(kind)
    root = check_root_safety(root, kind=kind)
    ensure_dir(root)
    ensure_dir(log_dir(root))
    print(f"[root] experiment-kind={kind}  output-root={root}")
    return root


def kind_note(kind: str, lambda1: float | None = None) -> str:
    if kind == "canonical_completion":
        return ("lambda_1 = canonical per dataset (losloop 0.02, shenzhen "
                "0.01); existing canonical graphs are COPIED here with "
                "provenance instead of being re-fitted")
    lam = "" if lambda1 is None else f" {lambda1:g}"
    return (f"lambda_1 ={lam} SUPPLEMENTARY sensitivity value (Los-loop "
            "only; canonical Los-loop is 0.02); everything in this tree is "
            "freshly computed — no canonical counterpart exists")
