"""
Generate the schematic figures for the manuscript.

These are drawn from code rather than a diagramming tool for the same reason
the numbers come from macros: the protocol changed once already, and hand-drawn
figures kept describing the superseded design. Anything asserted in a figure
here is asserted in one place.

Outputs (into results/):
    architecture.pdf   three-layer system view
    protocol.pdf       registration / proving / on-chain verification flow

Usage:
    python tools/make_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "text.usetex": False,
    "savefig.dpi": 300,
})

# Muted, print-safe, and distinguishable in greyscale by lightness.
C_REASON = "#dce6f2"
C_PROOF = "#fbe6d4"
C_CHAIN = "#dceadb"
C_SECRET = "#f4d7d7"
C_EDGE = "#4a4a4a"


def box(ax, x, y, w, h, text, fc, ec=C_EDGE, ls="-", fs=7.5, lw=0.9, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs, zorder=3,
        fontweight="bold" if bold else "normal", linespacing=1.45,
    )


def arrow(ax, p0, p1, style="-|>", color=C_EDGE, ls="-", lw=1.0, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=9,
            color=color, linewidth=lw, linestyle=ls, zorder=4,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def band(ax, y, h, label, fc):
    ax.add_patch(
        FancyBboxPatch(
            (0.008, y), 0.984, h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor=fc, edgecolor="none", alpha=0.4, zorder=0,
        )
    )
    ax.text(0.022, y + h - 0.028, label, fontsize=8, fontweight="bold",
            va="top", ha="left", color="#333333", zorder=1)


def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ======================================================================
# Figure 1: architecture
# ======================================================================

def figure_architecture(out: Path) -> None:
    fig, ax = new_ax(7.2, 4.6)

    band(ax, 0.68, 0.30, "KG Reasoning Layer  (model owner, private)", C_REASON)
    band(ax, 0.35, 0.30, "Proof Layer  (model owner, private)", C_PROOF)
    band(ax, 0.02, 0.30, "Ledger Layer  (Ethereum, public)", C_CHAIN)

    # --- reasoning ---
    box(ax, 0.05, 0.72, 0.19, 0.15,
        "Knowledge graph\n$\\mathcal{G}=(\\mathcal{E},\\mathcal{R},\\mathcal{T})$",
        C_REASON)
    box(ax, 0.29, 0.72, 0.21, 0.15,
        "Embedding model $\\theta$\nTransE / RotatE\nCompGCN / R-GCN", C_REASON)
    box(ax, 0.55, 0.72, 0.19, 0.15,
        "Prediction\n$(h,r,\\hat{t},s)$", C_REASON)
    box(ax, 0.79, 0.72, 0.16, 0.15,
        "weights digest\n$\\mathsf{wd}=\\mathrm{SHA256}(\\theta)$", C_SECRET, ls="--")

    arrow(ax, (0.24, 0.795), (0.29, 0.795))
    arrow(ax, (0.50, 0.795), (0.55, 0.795))
    arrow(ax, (0.50, 0.83), (0.79, 0.83), rad=-0.18)

    # --- proof ---
    box(ax, 0.05, 0.39, 0.24, 0.15,
        "Model commitment (once)\n$C_\\theta = vG + rH$\n$v=\\mathsf{H}(\\mathsf{wd}\\,\\|\\,\\mathsf{id})$",
        C_PROOF)
    box(ax, 0.345, 0.39, 0.22, 0.15,
        "Opening $(v,r)$\nnever transmitted", C_SECRET, ls="--")
    box(ax, 0.62, 0.39, 0.33, 0.15,
        "Schnorr-Fiat-Shamir proof\n"
        "$\\pi=(A,\\,s_v,\\,s_r,\\,\\mathsf{aux})$\n"
        "$e=\\mathsf{H}(C_\\theta\\|A\\|\\mathsf{ph}\\|\\mathsf{id})$",
        C_PROOF)

    arrow(ax, (0.87, 0.72), (0.17, 0.545), rad=0.10)
    arrow(ax, (0.645, 0.72), (0.78, 0.545), rad=-0.10)
    arrow(ax, (0.29, 0.465), (0.345, 0.465))
    arrow(ax, (0.565, 0.465), (0.62, 0.465))

    # --- chain ---
    box(ax, 0.05, 0.06, 0.24, 0.16,
        "Model registry\n$\\mathsf{id}\\mapsto(C_\\theta,\\ \\text{owner},\\ \\text{block})$\n"
        "single-shot", C_CHAIN, bold=False)
    box(ax, 0.345, 0.06, 0.28, 0.16,
        "On-chain verifier\ncheck  $s_vG+s_rH = A+eC_\\theta$\n"
        "precompiles $\\mathtt{0x06}/\\mathtt{0x07}$", C_CHAIN)
    box(ax, 0.68, 0.06, 0.27, 0.16,
        "Audit record\n$(\\mathsf{id},h,r,\\hat{t},s,\\mathsf{aux},\\mathsf{ph})$\n"
        "written only if valid", C_CHAIN)

    arrow(ax, (0.17, 0.39), (0.17, 0.22))
    arrow(ax, (0.78, 0.39), (0.485, 0.22), rad=0.10)
    arrow(ax, (0.29, 0.14), (0.345, 0.14))
    arrow(ax, (0.625, 0.14), (0.68, 0.14))
    ax.text(0.653, 0.232, "revert if invalid", fontsize=6.2, ha="center",
            va="bottom", color="#a03232", style="italic")

    ax.text(0.5, 0.985,
            "Dashed boxes never leave the model owner.",
            fontsize=7, ha="center", va="top", style="italic", color="#555555")

    fig.savefig(out / "architecture.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  -> {out / 'architecture.pdf'}")


# ======================================================================
# Figure 2: protocol
# ======================================================================

def figure_protocol(out: Path) -> None:
    fig, ax = new_ax(7.4, 5.2)

    # A left gutter holds the phase labels so they never collide with the
    # lane content; lanes are placed to the right of it.
    GUTTER = 0.105
    lane_x = [0.335, 0.615, 0.865]
    for x, name, fc in zip(
        lane_x,
        ["Model owner (prover)", "Consumer", "Ethereum contract"],
        [C_PROOF, C_REASON, C_CHAIN],
    ):
        box(ax, x - 0.115, 0.935, 0.23, 0.055, name, fc, bold=True, fs=7.8)
        ax.plot([x, x], [0.015, 0.935], color="#b0b0b0", lw=0.8,
                ls=(0, (4, 3)), zorder=0)

    def phase(y, text):
        ax.text(0.005, y, text, fontsize=7.2, style="italic",
                color="#666666", va="center", ha="left")

    def rule(y):
        ax.plot([0.005, 0.995], [y, y], color="#d5d5d5", lw=0.8)

    # ---- Phase 0: registration ----
    phase(0.845, "Phase 0\n(once, before\nserving)")
    box(ax, GUTTER + 0.01, 0.845, 0.44, 0.075,
        "$\\mathsf{wd}=\\mathrm{SHA256}(\\theta)$,    $r \\leftarrow \\mathbb{Z}_n$ (uniform)\n"
        "$C_\\theta = \\mathsf{H}(\\mathsf{wd}\\,\\|\\,\\mathsf{id})\\cdot G \\;+\\; r\\cdot H$",
        C_PROOF, fs=7.2)
    arrow(ax, (0.56, 0.800), (0.865, 0.800), lw=1.2)
    ax.text(0.712, 0.812,
            "$\\mathtt{registerModel}(\\mathsf{id},\\,C_\\theta,\\,\\mathsf{wd})$",
            fontsize=7, ha="center", va="bottom")
    box(ax, 0.715, 0.700, 0.28, 0.070,
        "store $(C_\\theta,\\ \\mathrm{sender},\\ \\mathrm{block})$\n"
        "reject re-registration", C_CHAIN, fs=7)

    rule(0.672)

    # ---- Phase 1: query and proof ----
    phase(0.545, "Phase 1\n(per query)")
    arrow(ax, (0.615, 0.635), (0.335, 0.635), lw=1.2)
    ax.text(0.475, 0.646, "query $(h, r, ?)$", fontsize=7.2,
            ha="center", va="bottom")

    box(ax, GUTTER + 0.01, 0.435, 0.50, 0.172,
        "$\\hat{t}=\\mathrm{argmax}_{t'}\\, f_\\theta(h,r,t')$,     $s=f_\\theta(h,r,\\hat{t})$\n"
        "$\\mathsf{aux}=\\mathsf{H}(\\mathbf{e}_h\\|\\mathbf{e}_r\\|\\mathbf{e}_{\\hat t})$\n"
        "$\\mathsf{ph}=\\mathsf{H}(\\mathsf{tag}\\|h\\|r\\|\\hat t\\|s\\|\\mathsf{aux})$\n"
        "$A=k_vG+k_rH$,     $e=\\mathsf{H}(C_\\theta\\|A\\|\\mathsf{ph}\\|\\mathsf{id})$\n"
        "$s_v=k_v+ev$,     $s_r=k_r+er$     (mod $n$)", C_PROOF, fs=7)

    arrow(ax, (0.335, 0.408), (0.615, 0.378), lw=1.2)
    ax.text(0.475, 0.352, "$(h,r,\\hat t,s)$,  $\\pi=(A,s_v,s_r,\\mathsf{aux})$",
            fontsize=7, ha="center", va="bottom")
    arrow(ax, (0.615, 0.345), (0.865, 0.322), lw=1.2)
    ax.text(0.740, 0.300, "$\\mathtt{verifyAndLog}(\\cdots)$",
            fontsize=7, ha="center", va="bottom")

    rule(0.300)

    # ---- Phase 2: on-chain verification ----
    phase(0.185, "Phase 2\n(consensus)")
    box(ax, 0.635, 0.078, 0.360, 0.195,
        "look up $C_\\theta$ in the registry\n"
        "check $A$ on curve;   $s_v, s_r < n$\n"
        "recompute $\\mathsf{ph}$ from the stored fields\n"
        "recompute $e$ from the transcript\n"
        "check   $s_vG + s_rH \\;=\\; A + e\\,C_\\theta$", C_CHAIN, fs=7)

    arrow(ax, (0.865, 0.038), (0.615, 0.038), lw=1.2)
    ax.text(0.740, 0.045, "record stored, or revert",
            fontsize=7, ha="center", va="bottom")

    ax.text(0.335, 0.175,
            "$C_\\theta$, $e$ and $\\mathsf{ph}$ are never\n"
            "transmitted. The contract\n"
            "derives all three itself.",
            fontsize=7.2, ha="center", va="center", style="italic",
            color="#a03232",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdf3f3",
                      edgecolor="#d9b3b3", linewidth=0.7))

    fig.savefig(out / "protocol.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  -> {out / 'protocol.pdf'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    print("Generating manuscript figures")
    figure_architecture(out)
    figure_protocol(out)


if __name__ == "__main__":
    main()
