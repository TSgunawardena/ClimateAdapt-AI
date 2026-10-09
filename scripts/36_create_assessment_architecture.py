"""Create the formal Assessment 2 architecture diagram and evidence summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "evidence" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

BUILT = "#d9ead3"
EXTERNAL = "#cfe2f3"
HUMAN = "#fce5cd"
ABSENT = "#e6e6e6"
EDGE = "#263238"


def box(ax, x, y, width, height, title, body, colour, linestyle="-"):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.6,
        edgecolor=EDGE,
        facecolor=colour,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.70,
        title,
        ha="center",
        va="center",
        fontsize=9.5,
        weight="bold",
        color="#17252a",
        zorder=3,
    )
    ax.text(
        x + width / 2,
        y + height * 0.35,
        body,
        ha="center",
        va="center",
        fontsize=7.6,
        color="#263238",
        linespacing=1.25,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, label=None, dashed=False, curve=0.0):
    style = "dashed" if dashed else "solid"
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.5,
            "color": "#455a64",
            "linestyle": style,
            "connectionstyle": f"arc3,rad={curve}",
        },
        zorder=1,
    )
    if label:
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            midpoint[0],
            midpoint[1] + 0.014,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#455a64",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            zorder=4,
        )


def create_diagram(output_png: Path, output_svg: Path) -> None:
    figure, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    figure.patch.set_facecolor("white")

    ax.text(
        0.5,
        0.965,
        "ClimateAdapt AI – Assessment 2 Prototype Architecture",
        ha="center",
        va="center",
        fontsize=19,
        weight="bold",
        color="#17324d",
    )
    ax.text(
        0.5,
        0.928,
        "Nelson coastal-flood susceptibility screening with constrained planning support",
        ha="center",
        va="center",
        fontsize=11,
        color="#455a64",
    )

    # Data and model-development plane.
    box(
        ax, 0.025, 0.735, 0.165, 0.135,
        "Authoritative data sources",
        "LINZ 1 m DEM + coastline\nLCDB v6 land cover\nNCC connected flood labels",
        EXTERNAL,
    )
    box(
        ax, 0.225, 0.735, 0.165, 0.135,
        "Geospatial preparation",
        "Geometry repair and coverage\n5 m alignment and rasterisation\nTraining-only normalisation",
        BUILT,
    )
    box(
        ax, 0.425, 0.735, 0.165, 0.135,
        "Prepared feature store",
        "Elevation · slope · coast distance\nLCDB one-hot channels · valid mask\nFixed geographic split masks",
        BUILT,
    )
    box(
        ax, 0.625, 0.735, 0.165, 0.135,
        "Model development and audit",
        "GroupNorm Compact U-Net V2\nLogistic + trivial baselines\nMC dropout · ablation · bias audit",
        BUILT,
    )
    box(
        ax, 0.825, 0.735, 0.15, 0.135,
        "Frozen model artefacts",
        "Best checkpoint (epoch 4)\nRocks Road F2 threshold 0.54\nFull-region output rasters",
        BUILT,
    )

    arrow(ax, (0.19, 0.802), (0.225, 0.802), "ingest")
    arrow(ax, (0.39, 0.802), (0.425, 0.802), "write")
    arrow(ax, (0.59, 0.802), (0.625, 0.802), "train/evaluate")
    arrow(ax, (0.79, 0.802), (0.825, 0.802), "freeze")

    # Serving and agent plane.
    box(
        ax, 0.025, 0.485, 0.165, 0.135,
        "User interfaces",
        "Streamlit evidence dashboard\nPoint/location request\nPlanning reviewer",
        BUILT,
    )
    box(
        ax, 0.225, 0.485, 0.165, 0.135,
        "Prediction HTTP service",
        "FastAPI /health · /model\nPOST /predict/point\nValidation + provenance contract",
        BUILT,
    )
    box(
        ax, 0.425, 0.485, 0.165, 0.135,
        "Prediction and guardrails",
        "Precomputed U-Net susceptibility\nUncertainty indicator\nPhysicalist terrain/connectivity check",
        BUILT,
    )
    box(
        ax, 0.625, 0.485, 0.165, 0.135,
        "Three-agent LangGraph",
        "1 Data validation agent\n2 Flood analysis agent\n3 Planning support agent",
        BUILT,
    )
    box(
        ax, 0.825, 0.485, 0.15, 0.135,
        "Local planning LLM",
        "Ollama runtime\nQwen3 4B Q4_K_M\nQLoRA fine-tuned adapter merge",
        EXTERNAL,
    )

    arrow(ax, (0.19, 0.552), (0.225, 0.552), "HTTP request")
    arrow(ax, (0.39, 0.552), (0.425, 0.552), "read cells")
    arrow(ax, (0.59, 0.552), (0.625, 0.552), "structured evidence")
    arrow(ax, (0.825, 0.552), (0.79, 0.552), "guarded JSON")
    arrow(ax, (0.90, 0.735), (0.51, 0.62), "serve frozen outputs", curve=0.14)

    # Decision and external-action plane.
    box(
        ax, 0.225, 0.245, 0.165, 0.125,
        "Human-review checkpoint",
        "Approve · revise · reject\nNo automatic release or action\nRevision limit and audit trail",
        HUMAN,
    )
    box(
        ax, 0.425, 0.245, 0.165, 0.125,
        "Controlled Gitea action",
        "Private local repository\nCreate issue only after approval\nEnvironment-only API token",
        EXTERNAL,
    )
    box(
        ax, 0.625, 0.245, 0.165, 0.125,
        "Independent verification",
        "Separate persisted-issue GET\nVerify title/body/run ID/approval\nEvidence JSON + audit log",
        BUILT,
    )
    box(
        ax, 0.825, 0.225, 0.15, 0.165,
        "Intentionally absent",
        "Hydrodynamic simulation\nLive tide/rain/drainage feeds\nEmergency-warning automation\nAutomatic regulatory decisions\nProduction cloud deployment",
        ABSENT,
        linestyle="--",
    )

    arrow(ax, (0.705, 0.485), (0.39, 0.31), "interrupt", curve=0.14)
    arrow(ax, (0.39, 0.307), (0.425, 0.307), "approved only")
    arrow(ax, (0.59, 0.307), (0.625, 0.307), "persist then verify")

    # Explicit replacement/deployment statement.
    box(
        ax, 0.025, 0.245, 0.165, 0.125,
        "Approved implementation choice",
        "Local Windows deployment\nLangGraph replaces OpenClaw\nLocal Ollama and Gitea retained\nDocument tutor-approved deviation",
        HUMAN,
    )

    # Evaluation strip.
    box(
        ax, 0.025, 0.065, 0.765, 0.09,
        "Evaluation evidence",
        (
            "Independent Tāhunanui test · base-vs-fine-tuned held-out comparison · "
            "17/17 robustness checks · 9/9 repeated composed-system cases · "
            "human-boundary and no-Gitea-mutation verification"
        ),
        BUILT,
    )
    box(
        ax, 0.825, 0.065, 0.15, 0.09,
        "Stopping condition",
        "Stop at human review unless\nauthorised approval is recorded",
        HUMAN,
    )

    legend = [
        Patch(facecolor=BUILT, edgecolor=EDGE, label="Built and evaluated in this project"),
        Patch(facecolor=EXTERNAL, edgecolor=EDGE, label="External/open-source or authoritative source"),
        Patch(facecolor=HUMAN, edgecolor=EDGE, label="Human-control or approved implementation choice"),
        Patch(facecolor=ABSENT, edgecolor=EDGE, label="Intentionally absent / out of scope"),
    ]
    ax.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )

    figure.savefig(output_png, dpi=220, bbox_inches="tight")
    figure.savefig(output_svg, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / "assessment2_system_architecture.png"
    svg = FIGURE_DIR / "assessment2_system_architecture.svg"
    create_diagram(png, svg)
    print("=" * 78)
    print("CLIMATEADAPT AI – ASSESSMENT 2 ARCHITECTURE")
    print("=" * 78)
    print(f"PNG: {png.relative_to(PROJECT_ROOT)}")
    print(f"SVG: {svg.relative_to(PROJECT_ROOT)}")
    print("RESULT: Architecture evidence generated.")


if __name__ == "__main__":
    main()
