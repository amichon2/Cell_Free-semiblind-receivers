"""Shared output settings for the figures.

Figures are written as PDF, not SVG.  pdflatex includes PDF directly with
\\includegraphics, so a document using them needs neither the `svg` package, nor
inkscape, nor `-shell-escape`.  The last point decides it: arXiv does not allow
`-shell-escape`, so a document built on \\includesvg cannot be compiled there.

matplotlib writes PDF natively, so this is not a conversion: the figures are vector
in both cases and nothing is resampled.

fonttype 42 embeds TrueType fonts instead of matplotlib's default Type 3, which both
arXiv and IEEE discourage (Type 3 subsets break text search and some PDF viewers).
"""
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42        # TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42

FIG_EXT = "pdf"                                  # extension for every report figure
