#!/usr/bin/env python3
"""
Create an easy-to-read progress story for the datanuri project (version 3).

This script keeps the numbers from the technical report, but retells them with
clear, everyday language. It prints the story to the console and saves the same
content to a text file next to this script.
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


DEFAULT_EXPORT_PATH = Path(__file__).resolve().parent / "report.txt"
HEADING_RULE = "-" * 24


def _wrap(text: str) -> str:
    return textwrap.fill(text, width=100)


def build_intro_section() -> str:
    lines: list[str] = []
    lines.append("What This Tool Does")
    lines.append(HEADING_RULE)
    lines.append(
        _wrap(
            "datanuri reads information from past brain cancer trials and gives two friendly answers: "
            "how many months someone like the patient usually stays stable, and the chance they are "
            "still doing well after six months. The goal is to give doctors and families a quick, "
            "grounded snapshot that supports their own judgement."
        )
    )
    lines.append("")
    lines.append(
        _wrap(
            "To teach the program, we fed it the cleaned research spreadsheet, asked it not to look at "
            "columns that only describe the studies, and saved age, sex, surgery success, daily activity "
            "scores, lab signals, and treatment history. It practices on most of the rows and we check its "
            "answers on the remaining rows it has never seen."
        )
    )
    lines.append("")
    return "\n".join(lines)


def build_quality_checks_section() -> str:
    lines: list[str] = []
    lines.append("How We Checked It")
    lines.append(HEADING_RULE)
    lines.append(
        _wrap(
            "The program repeats its study until the answers stop improving. Along the way we slow the "
            "learning if it races ahead, and we keep the best checkpoint. If the main program ever fails "
            "to load, older backups and even simple look-up rules step in so that the service still responds."
        )
    )
    lines.append("")
    return "\n".join(lines)


def build_results_section() -> str:
    lines: list[str] = []
    lines.append("How Well It Performed")
    lines.append(HEADING_RULE)
    lines.append(
        _wrap(
            "After practice the program faced a saved slice of trial records. The table shows how far each "
            "approach usually missed the real answer by (in months) and how often it correctly spotted a "
            "patient still doing well at the six-month mark."
        )
    )
    lines.append("")

    comparison_df = pd.DataFrame(
        [
            {
                "Program": "datanuri (v7a)",
                "Average miss (months)": 0.72,
                "Confidence score (0–1)": 0.93,
                "Six-month calls correct (%)": 90,
            },
            {
                "Program": "Straight-line guess",
                "Average miss (months)": 0.77,
                "Confidence score (0–1)": "",
                "Six-month calls correct (%)": "",
            },
            {
                "Program": "Gentle slope guess",
                "Average miss (months)": 1.08,
                "Confidence score (0–1)": "",
                "Six-month calls correct (%)": "",
            },
            {
                "Program": "Decision forest",
                "Average miss (months)": 1.06,
                "Confidence score (0–1)": "",
                "Six-month calls correct (%)": "",
            },
        ]
    )

    if TABULATE_AVAILABLE:
        lines.append(
            comparison_df.to_markdown(
                index=False,
                tablefmt="grid",
                floatfmt=".2f",
            )
        )
    else:
        lines.append(
            comparison_df.to_string(
                index=False,
                justify="left",
                float_format=lambda x: f"{x:.2f}",
            )
        )
        lines.append("(Install 'tabulate' for friendlier table borders: pip install tabulate)")
    lines.append("")

    lines.append(
        _wrap(
            "In short, datanuri offered the smallest average miss and is the only option above that still "
            "shares a confidence number for the six-month question."
        )
    )
    lines.append("")
    return "\n".join(lines)


def build_sample_cases_section() -> str:
    lines: list[str] = []
    lines.append("Sample Stories")
    lines.append(HEADING_RULE)
    lines.append(
        _wrap(
            "Below are three example patients. For each one we list the real time they remained steady, "
            "what datanuri guessed, and how that compares with a more traditional approach."
        )
    )
    lines.append("")

    holdout_df = pd.DataFrame(
        [
            {
                "Patient": "Example 1",
                "Real months stable": 1.9,
                "datanuri guess": 1.94,
                "Traditional guess": 2.07,
                "Chance of six-month wellness (%)": 38,
            },
            {
                "Patient": "Example 2",
                "Real months stable": 4.2,
                "datanuri guess": 4.28,
                "Traditional guess": 4.64,
                "Chance of six-month wellness (%)": 68,
            },
            {
                "Patient": "Example 3",
                "Real months stable": 7.3,
                "datanuri guess": 7.02,
                "Traditional guess": 9.01,
                "Chance of six-month wellness (%)": 80,
            },
        ]
    )

    if TABULATE_AVAILABLE:
        lines.append(
            holdout_df.to_markdown(
                index=False,
                tablefmt="grid",
                floatfmt=".2f",
            )
        )
    else:
        lines.append(
            holdout_df.to_string(
                index=False,
                justify="left",
                float_format=lambda x: f"{x:.2f}",
            )
        )
    lines.append("")
    lines.append(
        _wrap(
            "The final column shows the chance, expressed as a percentage, that each person is still doing "
            "well at six months. If the main program cannot supply the number, we fall back to a cautious "
            "estimate so no one is left without guidance."
        )
    )
    lines.append("")
    return "\n".join(lines)


def build_code_evaluation_section() -> str:
    lines: list[str] = []
    lines.append("How the Code Builds the Answers")
    lines.append(HEADING_RULE)
    steps = [
        (
            "1. Gather and tidy the data",
            "We start by opening the cleaned trial spreadsheet. Columns that only describe the study "
            "are removed. We keep the details that describe the person: age, sex, surgery success, daily "
            "activity score, lab signal, and treatment history. We also make sure the two goal columns "
            "we care about are present."
        ),
        (
            "2. Set aside a fair test",
            "Most rows are used for practice, while a smaller slice is hidden away for checking later. "
            "When the yes/no six-month result is available, we shuffle the rows in a way that keeps the "
            "same balance of successes in both piles so the check feels fair."
        ),
        (
            "3. Put numbers on the same scale",
            "Some features are big numbers, some are small. We teach the program to centre and scale the "
            "inputs using only the practice rows, and we reuse the same scaler later so the service speaks "
            "the same language as the training run."
        ),
        (
            "4. Teach the brain",
            "The model itself is a stack of layers that share information, then split into two heads. "
            "One head guesses months, the other gives the six-month chance. Dropout and other guard rails "
            "keep the network from memorising instead of learning."
        ),
        (
            "5. Keep the best version",
            "During training we nudge the learning rate, stop early if progress stalls, and only save the "
            "best checkpoint. A temperature control keeps the six-month confidence number calm and realistic."
        ),
        (
            "6. Check the work",
            "After training we run the model on the saved-away slice. We measure how many months off the "
            "guess was on average and how often it called the six-month outcome right. The tables in this "
            "report are generated from those numbers."
        ),
        (
            "7. Share the answers safely",
            "When the service runs, it loads the saved model and scaler, applies a small clinical adjustment "
            "to the six-month chance, and serves the `/predict` and `/debug-predict` endpoints. If something "
            "goes wrong, older checkpoints or simple fallbacks step in so the user still gets guidance."
        ),
    ]
    for title, paragraph in steps:
        lines.append(title)
        lines.append("")
        lines.append(_wrap(paragraph))
        lines.append("")
    return "\n".join(lines)


def build_closing_section() -> str:
    lines: list[str] = []
    lines.append("Takeaway")
    lines.append(HEADING_RULE)
    lines.append(
        _wrap(
            "datanuri brings the careful math into a plain-language summary. It gives a quick, calm starting "
            "point for conversations, while doctors continue to make the final call."
        )
    )
    lines.append("")
    return "\n".join(lines)


def build_report() -> str:
    sections = [
        build_intro_section(),
        build_quality_checks_section(),
        build_results_section(),
        build_code_evaluation_section(),
        build_sample_cases_section(),
        build_closing_section(),
    ]
    return "\n".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a plain-language datanuri summary (v3)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPORT_PATH,
        help=f"Path to save the story (default: {DEFAULT_EXPORT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_text = build_report()

    print(report_text)

    export_path = args.output
    export_path.write_text(report_text, encoding="utf-8")
    print(f"\nStory saved to: {export_path.resolve()}")


if __name__ == "__main__":
    main()

