"""Flatten revised_version sections into sn-article-flat.tex."""
from pathlib import Path

base = Path(__file__).resolve().parent
src = (base / "sn-article.tex").read_text(encoding="utf-8")

# Front matter: from \title through \maketitle
t0 = src.index("\\title[")
t1 = src.index("\\maketitle") + len("\\maketitle")
front = src[t0:t1]

# Intro/Background: after maketitle until first section input of method
m0 = src.index("\\maketitle") + len("\\maketitle")
m1 = src.index("\\input{sections/method}")
introbg = src[m0:m1]

parts = []
for name in [
    "method.tex",
    "setup.tex",
    "results.tex",
    "discussion_limitations_conclusion.tex",
]:
    parts.append((base / "sections" / name).read_text(encoding="utf-8"))

appB = (base / "sections" / "appendix_bibliometric.tex").read_text(encoding="utf-8")
appM = (base / "sections" / "appendix_mae.tex").read_text(encoding="utf-8")

preamble = r"""% ======================================================================
% FLATTENED revised manuscript (submission-oriented)
% Sections inlined; compile in this directory with pdflatex/bibtex
% ======================================================================
\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}

\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{mathrsfs}
\usepackage[title]{appendix}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{manyfoot}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{subcaption}

\theoremstyle{thmstyleone}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\theoremstyle{thmstyletwo}
\newtheorem{example}{Example}
\newtheorem{remark}{Remark}
\theoremstyle{thmstylethree}
\newtheorem{definition}{Definition}

\raggedbottom
\input{commands}

\begin{document}

"""

out = preamble + front + "\n" + introbg + "\n"
out += "\n\n".join(parts) + "\n"
out += "\\begin{appendices}\n" + appB + "\n" + appM + "\n\\end{appendices}\n"
out += "\n\\bibliography{MyReferences}\n\n\\end{document}\n"

flat = base / "sn-article-flat.tex"
flat.write_text(out, encoding="utf-8")
print("Wrote", flat, "bytes", flat.stat().st_size)
