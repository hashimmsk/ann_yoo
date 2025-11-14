#!/usr/bin/env python3
"""
Generate a formatted methods/results summary for the ADJANN project and export it.

Running this script prints the method description plus the comparative results tables
and writes the same content to an exportable text file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import pandas as pd
from pandas.io.formats.style import Styler


try:
    import tabulate  # type: ignore  # noqa: F401
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False


DEFAULT_EXPORT_PATH = Path("generated_report.txt")
HEADING_RULE = "-" * 16


def build_method_logic_section() -> str:
    lines: list[str] = []
    lines.append("Method & Logic")
    lines.append(HEADING_RULE)
    bullets = [
        "Multi-task neural network: shared dense trunk with dropout/batch norm feeding two heads - "
        "linear mPFS regressor and temperature-scaled sigmoid PFS6 classifier - trained jointly with "
        "Adam, equal loss weighting, and task-specific metrics for clinical interpretability.",
        "Data pipeline: starts from trial-level CSV, drops metadata columns, performs a stratified "
        "80/20 split on binary PFS6, fits a StandardScaler on the train fold only, and uses callbacks "
        "(EarlyStopping, ReduceLROnPlateau, ModelCheckpoint) to limit overfitting while preserving the "
        "best weights.",
        "Program logic: FastAPI backend loads the latest v7a artifacts, applies calibrated inference "
        "(age/performance/resection heuristics) when returning PFS6 percentages, exposes raw "
        "probabilities for transparency, and falls back to older models or heuristic estimates if "
        "neural predictions fail."
    ]

    wrapper_width = 100
    lines.append("")
    for bullet in bullets:
        wrapped = textwrap.fill(bullet, width=wrapper_width)
        for idx, segment in enumerate(wrapped.splitlines()):
            prefix = "  - " if idx == 0 else "    "
            lines.append(f"{prefix}{segment}")
    lines.append("")
    return "\n".join(lines)


def build_results_section() -> str:
    lines: list[str] = []
    lines.append("Results")
    lines.append(HEADING_RULE)
    lines.append(
        textwrap.fill(
            "Models were benchmarked on a held-out validation fold after training on the remaining "
            "data. The table summarizes regression error (mPFS) and classification performance "
            "(PFS6) for the multi-task ANN compared with classical baselines.",
            width=100,
        )
    )
    lines.append("")

    comparison_df = pd.DataFrame(
        [
            {
                "Model": "AJDANN v7a (multitask ANN)",
                "RMSE": 1.3446,
                "MAE": 1.1391,
                "AUC": 1.0000,
                "ACC": 1.0000,
            },
            {
                "Model": "Linear Regression",
                "RMSE": 1.0242,
                "MAE": 0.7723,
                "AUC": float("nan"),
                "ACC": float("nan"),
            },
            {
                "Model": "Ridge Regression (alpha=1)",
                "RMSE": 1.0910,
                "MAE": 1.0837,
                "AUC": float("nan"),
                "ACC": float("nan"),
            },
            {
                "Model": "Random Forest Regressor",
                "RMSE": 1.1015,
                "MAE": 1.0555,
                "AUC": float("nan"),
                "ACC": float("nan"),
            },
        ]
    )

    lines.append("Validation RMSE / MAE / Classification Metrics")
    if TABULATE_AVAILABLE:
        lines.append(comparison_df.to_markdown(index=False, tablefmt="grid", floatfmt=".4f"))
    else:
        lines.append(comparison_df.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}"))
        lines.append(
            "(Install 'tabulate' for grid formatting: pip install tabulate)"
        )
    lines.append("")

    return "\n".join(lines)


def build_validation_section() -> str:
    lines: list[str] = []
    lines.append("Validation")
    lines.append(HEADING_RULE)
    lines.append(
        textwrap.fill(
            "An 80/20 stratified split on PFS6 produced three holdout trials for sanity checking. "
            "The table reports actual mPFS alongside predictions from each model; ANN probabilities "
            "are calibrated in the FastAPI layer before being shown to clinicians.",
            width=100,
        )
    )
    lines.append("")

    holdout_df = pd.DataFrame(
        [
            {
                "Trial": "Sample 1",
                "Actual mPFS": 1.9,
                "AJDANN v7a": 2.4386,
                "Random Forest": 3.3662,
                "Linear Regression": 2.0690,
            },
            {
                "Trial": "Sample 2",
                "Actual mPFS": 4.2,
                "AJDANN v7a": 3.4642,
                "Random Forest": 5.1996,
                "Linear Regression": 4.6369,
            },
            {
                "Trial": "Sample 3",
                "Actual mPFS": 7.3,
                "AJDANN v7a": 5.1570,
                "Random Forest": 6.5994,
                "Linear Regression": 9.0109,
            },
        ]
    )

    lines.append("Holdout Predictions (mPFS, months)")
    if TABULATE_AVAILABLE:
        lines.append(holdout_df.to_markdown(index=False, tablefmt="grid", floatfmt=".4f"))
    else:
        lines.append(
            holdout_df.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}")
        )
        lines.append(
            "(Install 'tabulate' for grid formatting: pip install tabulate)"
        )
    lines.append("")

    return "\n".join(lines)


def build_report() -> str:
    method_section = build_method_logic_section()
    results_section = build_results_section()
    validation_section = build_validation_section()
    return f"{method_section}\n{results_section}\n{validation_section}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and export the ADJANN methods/results report."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPORT_PATH,
        help=f"Path to write the exportable report (default: {DEFAULT_EXPORT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_text = build_report()

    print(report_text)

    export_path = args.output
    export_path.write_text(report_text, encoding="utf-8")
    print(f"\nReport exported to: {export_path.resolve()}")


if __name__ == "__main__":
    main()

