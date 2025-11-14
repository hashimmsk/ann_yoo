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
        "Multitask dense neural network predicts mean progression-free survival (mPFS) as a regression "
        "target and six-month PFS (PFS6) as a calibrated probability, using dropout and batch "
        "normalization in the shared trunk before the linear and temperature-scaled heads; training "
        "uses Adam with balanced task losses for clinical interpretability.",
        "Training pipeline loads the curated trial dataset, removes metadata columns so that age, sex, "
        "resection percentage, Karnofsky score, methylation, and treatment history remain as inputs, "
        "performs a stratified 80/20 split on binarized PFS6, fits a StandardScaler on the training "
        "fold only, and applies EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks.",
        "Serving layer uses a FastAPI backend that loads the latest model artifacts, calibrates PFS6 "
        "probabilities with age/performance/resection heuristics before returning percentages, exposes "
        "raw probabilities for transparency, and falls back to cached models or heuristic estimates if "
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
    lines.append(
        textwrap.fill(
            "Key outcomes: AJDANN v7a achieves the lowest mPFS error (~7% reduction in RMSE/MAE vs. "
            "the best baseline) and is the only model that provides calibrated PFS6 probabilities with "
            "good discrimination (AUC 0.93, ACC 0.90).",
            width=100,
        )
    )
    lines.append("")

    comparison_df = pd.DataFrame(
        [
            {
                "Model": "AJDANN v7a (multitask ANN)",
                "RMSE": 0.9580,
                "MAE": 0.7150,
                "AUC": 0.9270,
                "ACC": 0.9000,
            },
            {
                "Model": "Linear Regression",
                "RMSE": 1.0242,
                "MAE": 0.7723,
                "AUC": "--",
                "ACC": "--",
            },
            {
                "Model": "Ridge Regression (alpha=1)",
                "RMSE": 1.0910,
                "MAE": 1.0837,
                "AUC": "--",
                "ACC": "--",
            },
            {
                "Model": "Random Forest Regressor",
                "RMSE": 1.1015,
                "MAE": 1.0555,
                "AUC": "--",
                "ACC": "--",
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
                "AJDANN v7a": 1.9420,
                "Random Forest": 3.3662,
                "Linear Regression": 2.0690,
            },
            {
                "Trial": "Sample 2",
                "Actual mPFS": 4.2,
                "AJDANN v7a": 4.2845,
                "Random Forest": 5.1996,
                "Linear Regression": 4.6369,
            },
            {
                "Trial": "Sample 3",
                "Actual mPFS": 7.3,
                "AJDANN v7a": 7.0218,
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

    # PFS6 probability table
    pfs6_table = pd.DataFrame(
        [
            {
                "Trial": "Sample 1",
                "PFS6_true (%)": 43.0,
                "AJDANN v7a (raw prob)": 0.3780,
                "AJDANN v7a (calibrated %)": 37.8,
                "Calibrated range": "~35-45%",
            },
            {
                "Trial": "Sample 2",
                "PFS6_true (%)": 58.0,
                "AJDANN v7a (raw prob)": 0.8750,
                "AJDANN v7a (calibrated %)": 68.0,
                "Calibrated range": "~50-65%",
            },
            {
                "Trial": "Sample 3",
                "PFS6_true (%)": 72.0,
                "AJDANN v7a (raw prob)": 0.9980,
                "AJDANN v7a (calibrated %)": 80.0,
                "Calibrated range": "~70-80%",
            },
        ]
    )

    lines.append("Holdout Predictions (PFS6, percentages)")
    if TABULATE_AVAILABLE:
        lines.append(pfs6_table.to_markdown(index=False, tablefmt="grid", floatfmt=".4f"))
    else:
        lines.append(
            pfs6_table.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}")
        )
        lines.append(
            "(Install 'tabulate' for grid formatting: pip install tabulate)"
        )
    lines.append("")

    lines.append(
        textwrap.fill(
            "Taken together, these results indicate that the project successfully delivered an "
            "end-to-end research tool (datanuri) that improves mPFS prediction over classical "
            "baselines while providing calibrated, clinically interpretable PFS6 estimates via a "
            "robust API.",
            width=100,
        )
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

