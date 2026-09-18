#!/usr/bin/env python
"""Regenerate every published benchmark number from the pipeline itself.

Why this exists
---------------
The README and the Streamlit app quote the same experiment, but they were
generated from two different datasets: the README table came from the full
59,946-row extract, while the app is hard-wired to the committed 10,000-row
sample. Nothing in either file said so, and the numbers were typed in by hand,
so they drifted the moment the code changed.

This script trains the real pipeline (``smoking_model.train_model_suite``) on a
chosen dataset and writes the resulting table into the marked block in
``README.md``, plus a full two-dataset report in ``RESULTS.md``. Numbers written
by this script cannot drift from the code, because they *are* the code's output.

Usage (from the repository root)::

    python scripts/report_metrics.py                      # print every dataset that is present
    python scripts/report_metrics.py --dataset sample      # print one
    python scripts/report_metrics.py --write               # regenerate README block + RESULTS.md

The ``full`` dataset is git-ignored, so on a fresh clone only ``sample`` is
available; that is expected and reported rather than treated as an error.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ``python scripts/report_metrics.py`` puts scripts/ on sys.path, not the repo
# root, so the project modules are not importable without this. Same class of
# problem as the one that broke CI; see conftest.py.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, recall_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from data_loader import load_data  # noqa: E402
from smoking_model import SMOKER_POSITIVE_LABELS, train_model_suite  # noqa: E402

# The label mapping the analysis notebook uses for its target column.
SMOKE_MAP = {"no": 0, "sometimes": 1, "when drinking": 1, "yes": 1, "trying to quit": 1}
FEATURES = ["drugs", "body_type", "age", "drinks"]

SAMPLE_PATH = os.path.join(REPO_ROOT, "data", "sample_profiles.csv")
FULL_PATH = os.path.join(REPO_ROOT, "data", "full", "profiles.csv")

DATASETS = {
    "sample": {
        "heading": "Committed sample — 10,000 rows",
        "blurb": (
            "shipped in the repository and served by the deployed app. Stratified on sex "
            "and smoking status, so shares stay representative."
        ),
        "path": SAMPLE_PATH,
    },
    "full": {
        "heading": "Full 2012 extract — 59,946 rows",
        "blurb": (
            "the complete Codecademy extract, git-ignored rather than rehosted "
            "(see DATA.md). The analysis notebook was run on this dataset."
        ),
        "path": FULL_PATH,
    },
}

# Display names for the README, kept human-readable rather than the dict keys
# used internally by smoking_model.
MODEL_LABELS = {
    "Baseline (Always 'No')": 'Baseline — always answer "does not smoke"',
    "Logistic Regression (Unweighted)": "Logistic regression",
    "Logistic Regression (Balanced)": 'Logistic regression, `class_weight="balanced"`',
    "Random Forest (Balanced, max_depth=5)": "Random forest, `max_depth=5`",
    "HistGradientBoosting (Balanced)": "HistGradientBoosting (balanced)",
}

COLUMNS = ["Accuracy", "Precision (smoker)", "Recall (smoker)", "F1 (smoker)", "ROC-AUC"]
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def analyse(dataset: str) -> dict | None:
    """Train the suite on one dataset and return everything the report needs."""
    spec = DATASETS[dataset]
    if not os.path.exists(spec["path"]):
        return None

    os.environ["OKCUPID_CSV"] = spec["path"]
    df = load_data()
    suite = train_model_suite(df)

    answered = int(suite["X_train_shape"][0] + suite["X_test_shape"][0])
    positive = df["smokes"].isin(SMOKER_POSITIVE_LABELS)

    # The "did not answer" versus "answered never" effect that motivates
    # treating missing values as an explicit category.
    rated = df[df["smokes"].isin(SMOKER_POSITIVE_LABELS | {"no"})]
    by_drugs = rated.groupby("drugs")["smokes"].apply(lambda s: s.isin(SMOKER_POSITIVE_LABELS).mean())

    return {
        "dataset": dataset,
        "heading": spec["heading"],
        "blurb": spec["blurb"],
        "rows": len(df),
        "answered": answered,
        "test_rows": int(suite["X_test_shape"][0]),
        "base_rate": float(suite["base_rate"]),
        "metrics": suite["metrics"],
        "impact": suite["feature_impact_df"],
        "never_drugs_rate": float(by_drugs.get("never", float("nan"))),
        "blank_drugs_rate": float(by_drugs.get("unspecified", float("nan"))),
        "blank_drugs_rows": int((rated["drugs"] == "unspecified").sum()),
        "smoker_rows": int(positive.sum()),
    }


def encoding_ablation() -> dict | None:
    """Measure what the explicit ``unspecified`` category is worth.

    The analysis notebook encodes with ``get_dummies(drop_first=True)`` on the raw
    columns, so an unanswered question produces an all-zero row that is
    indistinguishable from the dropped reference level. The shipped pipeline gives
    missing values their own category instead.

    This rebuilds the notebook's encoding on the same rows, the same features and the
    same split as the shipped pipeline, so the only difference between the two sets of
    numbers is how a blank answer is encoded. Requires the full extract, which is not
    committed.
    """
    if not os.path.exists(FULL_PATH):
        return None

    df = pd.read_csv(FULL_PATH, usecols=["smokes", *FEATURES])
    df["smokes_binary"] = df["smokes"].map(SMOKE_MAP)
    df = df.dropna(subset=["smokes_binary"])

    X = pd.get_dummies(df[FEATURES], columns=["drugs", "body_type", "drinks"], drop_first=True)
    y = df["smokes_binary"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scores = {}
    for label, model in (
        ("unweighted", LogisticRegression(max_iter=1000)),
        ("balanced", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        scores[label] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "recall": float(recall_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds)),
        }

    drugs_block = [name for name in X.columns if name.startswith("drugs_")]
    return {
        "columns": int(X.shape[1]),
        "rows": int(len(X)),
        "ambiguous_drug_rows": int((X[drugs_block].sum(axis=1) == 0).sum()),
        "scores": scores,
    }


def markdown_table(report: dict) -> str:
    """The benchmark table for one dataset, as Markdown."""
    lines = [
        "| Model | " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * (len(COLUMNS) + 1),
    ]
    for name, m in report["metrics"].items():
        label = MODEL_LABELS.get(name, name)
        cells = " | ".join(f"{m[key]:.3f}" for key in METRIC_KEYS)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def context_block(report: dict) -> str:
    """Rows, split size and base rate — the numbers that say *which* run this is."""
    return (
        f"- Profiles in file: **{report['rows']:,}**\n"
        f"- Profiles with a usable smoking label: **{report['answered']:,}**\n"
        f"- Held-out test split (20%, stratified): **{report['test_rows']:,}**\n"
        f"- Smoker base rate: **{report['base_rate']:.1%}**\n"
        f"- Smoker rate when the drugs question was answered `never`: "
        f"**{report['never_drugs_rate']:.1%}** ({report['answered'] - report['blank_drugs_rows']:,} profiles)\n"
        f"- Smoker rate when the drugs question was left blank: "
        f"**{report['blank_drugs_rate']:.1%}** ({report['blank_drugs_rows']:,} profiles)"
    )


def readme_block(report: dict) -> str:
    return (
        f"**{report['heading']}** — {report['blurb']}\n\n"
        f"{markdown_table(report)}\n\n"
        f"{context_block(report)}\n\n"
        "_Generated by `python scripts/report_metrics.py --write`; do not edit by hand._"
    )


def write_readme(report: dict) -> bool:
    """Replace the marked benchmark block in README.md in place."""
    path = os.path.join(REPO_ROOT, "README.md")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    start, end = "<!-- benchmark:start -->", "<!-- benchmark:end -->"
    if start not in text or end not in text:
        print(f"! {start} / {end} markers not found in README.md; skipped.")
        return False

    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    updated = f"{head}{start}\n{readme_block(report)}\n{end}{tail}"

    if updated == text:
        print("= README.md benchmark block already current.")
        return False
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(updated)
    print("+ README.md benchmark block regenerated.")
    return True


def coefficient_table(report: dict) -> str:
    """The balanced logistic regression's most influential features, with odds ratios."""
    impact = report["impact"].copy()
    ordered = pd.concat([impact.head(5), impact.tail(5)]).sort_values(
        "lr_balanced_coef", ascending=False
    )
    lines = ["| Feature | Coefficient (log-odds) | Odds ratio |", "|---|---|---|"]
    for _, row in ordered.iterrows():
        lines.append(
            f"| `{row['feature']}` | {row['lr_balanced_coef']:+.3f} | "
            f"{row['lr_balanced_odds_ratio']:.2f}x |"
        )
    return "\n".join(lines)


def ablation_table(ablation: dict, full: dict | None) -> str:
    """Notebook encoding versus shipped encoding, side by side."""
    lines = [
        "| Encoding | Columns | Model | Accuracy | Recall | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for label, scores in ablation["scores"].items():
        lines.append(
            f"| Notebook — blanks folded into the reference | {ablation['columns']} | "
            f"{label} | {scores['accuracy']:.3f} | {scores['recall']:.3f} | {scores['f1']:.3f} |"
        )
    if full is not None:
        for name, suffix in (
            ("Logistic Regression (Unweighted)", "unweighted"),
            ("Logistic Regression (Balanced)", "balanced"),
        ):
            m = full["metrics"][name]
            lines.append(
                "| Shipped — explicit `unspecified` category | "
                f"{len(full['impact'])} | {suffix} | {m['accuracy']:.3f} | {m['recall']:.3f} | "
                f"{m['f1']:.3f} |"
            )
    return "\n".join(lines)


def write_results(reports: list[dict], ablation: dict | None) -> None:
    """Write RESULTS.md holding every dataset that could be read."""
    path = os.path.join(REPO_ROOT, "RESULTS.md")
    parts = [
        "# Model results",
        "",
        "Every figure below is produced by `scripts/report_metrics.py`, which trains the",
        "pipeline in `smoking_model.py` and prints what it actually measured. The headline",
        "point of the table is the comparison between the baseline and the models: on",
        "imbalanced data, accuracy alone is not evidence of anything.",
        "",
        "The repository ships a 10,000-row sample and git-ignores the full extract, so the",
        "two datasets below give slightly different numbers. Both are correct; quoting one",
        "without naming it is not, which is why every table is labelled.",
        "",
    ]
    for report in reports:
        parts += [
            f"## {report['heading']}",
            "",
            # Repo-relative on purpose: the generated file is committed, and an absolute
            # path would publish the developer's home directory.
            "*reads `"
            f"{os.path.relpath(DATASETS[report['dataset']]['path'], REPO_ROOT).replace(os.sep, '/')}`*",
            "",
            markdown_table(report),
            "",
            context_block(report),
            "",
            "### What drives the balanced model",
            "",
            "Coefficients from the balanced logistic regression, largest positive and",
            "largest negative. Odds ratios above 1 increase the odds of the profile being",
            "labelled a smoker, below 1 decrease them.",
            "",
            coefficient_table(report),
            "",
        ]
    if ablation is not None:
        full = next((r for r in reports if r["dataset"] == "full"), None)
        rate = ablation["ambiguous_drug_rows"] / ablation["rows"]
        parts += [
            "## Why the notebook's numbers differ from the shipped pipeline's",
            "",
            "The notebook encodes its features with `pd.get_dummies(..., drop_first=True)` on",
            "the raw columns, so an unanswered question becomes an all-zero row that the model",
            "cannot tell apart from the dropped reference level (`never` for drugs). The",
            "shipped pipeline gives blank answers their own `unspecified` category instead.",
            "",
            "Same rows, same four features, same split and seed — only the encoding differs:",
            "",
            ablation_table(ablation, full),
            "",
            f"Under the notebook's encoding, {ablation['ambiguous_drug_rows']:,} of "
            f"{ablation['rows']:,} rows ({rate:.0%}) encode identically for \u201creports never",
            "using drugs\u201d and \u201cdid not answer the question\u201d, so the distinction measured",
            "directly above is invisible to the model. Giving blanks their own category",
            f"recovers it: balanced-model recall moves {ablation['scores']['balanced']['recall']:.3f} "
            f"to {full['metrics']['Logistic Regression (Balanced)']['recall']:.3f} on the same "
            f"split, with accuracy moving "
            f"{ablation['scores']['balanced']['accuracy']:.3f} to "
            f"{full['metrics']['Logistic Regression (Balanced)']['accuracy']:.3f}. That is one "
            "split rather than cross-validation, so read the direction as the finding and the "
            "size of the gap as an estimate.",
            "",
        ]
    parts += [
        "## Reading the table",
        "",
        f"About {1 - report['base_rate']:.1%} of profiles with a usable label do not smoke, so the",
        "baseline row is not a weak model — it is the number to beat. The unweighted logistic",
        "regression edges past it on accuracy while catching a small minority of smokers;",
        "balancing the class weights gives back a large amount of recall and gives up accuracy",
        "in exchange. Which of those two rows is \"better\" depends entirely on whether the cost",
        "of a missed smoker or a false alarm is higher, and that is a product decision rather",
        "than a statistical one.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python scripts/report_metrics.py --write",
        "```",
        "",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(parts))
    print(f"+ RESULTS.md written ({len(reports)} dataset(s)).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        choices=["sample", "full", "both"],
        default="both",
        help="which dataset to run (default: both, skipping any that is absent)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate the README.md benchmark block and RESULTS.md",
    )
    args = parser.parse_args()

    wanted = list(DATASETS) if args.dataset == "both" else [args.dataset]
    reports: list[dict] = []
    for name in wanted:
        report = analyse(name)
        if report is None:
            print(f"- {name}: not present at {DATASETS[name]['path']} — skipped.")
            continue
        reports.append(report)
        print(f"\n== {report['heading']} ==")
        print(markdown_table(report))
        print(context_block(report))

    if not reports:
        print("No dataset available; nothing to report.", file=sys.stderr)
        return 1

    ablation = encoding_ablation()
    if ablation is None:
        print("\n- encoding ablation: needs the full extract, skipped.")
    else:
        print("\n== Encoding ablation: notebook vs shipped ==")
        print(ablation_table(ablation, next((r for r in reports if r["dataset"] == "full"), None)))

    if args.write:
        # The README block is the sample run, because that is what the deployed app
        # serves; RESULTS.md carries every dataset the machine can read.
        sample = next((r for r in reports if r["dataset"] == "sample"), None)
        if sample is None:
            print("! README block left alone: it documents the sample run.", file=sys.stderr)
        else:
            write_readme(sample)
        write_results(reports, ablation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
