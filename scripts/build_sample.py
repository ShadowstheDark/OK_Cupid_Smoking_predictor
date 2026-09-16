#!/usr/bin/env python3
"""Build the small, committable demo sample used by the Streamlit app.

The full OkCupid extract (``profiles.csv``) is ~151 MB, which is over GitHub's
100 MB per-file limit, so it is never committed. This script produces a
stratified, essay-truncated subset instead.

It deliberately uses **only the standard library**, so anybody who clones the
repo can regenerate the sample without installing pandas first::

    python scripts/build_sample.py
    python scripts/build_sample.py --rows 5000 --seed 7 --max-essay-chars 600

Why stratified rather than a plain random sample
------------------------------------------------
Rows are sampled proportionally within each ``(sex, smokes)`` group. That keeps
two things representative at once:

* the app's demographic breakdowns and 12 astrological signs stay populated;
* the notebook's ~81% / 19% non-smoker / smoker split survives, so the
  class-imbalance story still reads correctly on the sample.

A plain random sample of a few thousand rows would leave the Match Calculator
returning "Zero Matches Found" for ordinary criteria (it chains twelve filters
together) and would distort the smoker base rate.

On file size
------------
The ten ``essayN`` columns are ~85% of the bytes. Truncating them much harder
would shrink the sample further, but it would also break the Profile Detective
page, which shows real bios and searches them by keyword. The default of 400
characters keeps the long tail in check (the app only ever displays 280) while
leaving the committed file around 17 MB -- comfortably inside GitHub's 100 MB
per-file limit.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "full" / "profiles.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "sample_profiles.csv"

# Groups the sample is kept proportional across.
STRATIFY_ON = ("sex", "smokes")

# Truncated in the sample only, to keep the committed file small.
ESSAY_COLUMNS = tuple(f"essay{i}" for i in range(10))

TRUNCATION_MARKER = "\u2026"


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def iter_body(path: Path, expected_width: int):
    """Yield data rows, skipping any malformed line with the wrong width."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # discard header
        for row in reader:
            if len(row) == expected_width:
                yield row


def count_strata(path: Path, header: list[str]) -> tuple[Counter, int]:
    """First pass: how many rows exist in each (sex, smokes) group."""
    positions = [header.index(column) for column in STRATIFY_ON]
    counts: Counter = Counter()
    for row in iter_body(path, len(header)):
        counts[tuple(row[i] for i in positions)] += 1
    return counts, sum(counts.values())


def allocate_quotas(counts: Counter, target: int) -> dict:
    """Split ``target`` across strata proportionally, using largest remainder.

    Never asks a stratum for more rows than it actually has, and redistributes
    the shortfall created by that clamping.
    """
    total = sum(counts.values())
    if total == 0 or target <= 0:
        return {key: 0 for key in counts}

    exact = {key: count * target / total for key, count in counts.items()}
    quotas = {key: min(int(math.floor(value)), counts[key]) for key, value in exact.items()}

    def headroom(key: str) -> int:
        return counts[key] - quotas[key]

    # Hand out what the flooring left over, to the largest fractional parts.
    deficit = target - sum(quotas.values())
    for key in sorted(exact, key=lambda k: (exact[k] - math.floor(exact[k]), counts[k]), reverse=True):
        if deficit <= 0:
            break
        give = min(headroom(key), 1)
        quotas[key] += give
        deficit -= give

    # If clamping below created a bigger shortfall, top up the largest strata.
    while deficit > 0:
        progress = False
        for key in sorted(counts, key=lambda k: headroom(k), reverse=True):
            if deficit <= 0:
                break
            give = min(headroom(key), deficit)
            if give > 0:
                quotas[key] += give
                deficit -= give
                progress = True
        if not progress:
            break

    return quotas


def stratified_sample(path: Path, header: list[str], quotas: dict, rng: random.Random) -> list[list[str]]:
    """Second pass: reservoir-sample each stratum down to its quota (Algorithm R)."""
    positions = [header.index(column) for column in STRATIFY_ON]
    reservoirs: dict = {key: [] for key in quotas}
    seen: Counter = Counter()

    for row in iter_body(path, len(header)):
        key = tuple(row[i] for i in positions)
        quota = quotas.get(key, 0)
        if quota <= 0:
            continue

        seen[key] += 1
        reservoir = reservoirs[key]
        if len(reservoir) < quota:
            reservoir.append(row)
        else:
            # For the i-th item seen, swap it in with probability quota / i.
            index = rng.randint(0, seen[key] - 1)
            if index < quota:
                reservoir[index] = row

    return [row for key in sorted(reservoirs) for row in reservoirs[key]]


def truncate_essays(row: list[str], essay_positions: list[int], max_chars: int | None) -> list[str]:
    if not max_chars or max_chars <= 0:
        return row
    trimmed = list(row)
    for position in essay_positions:
        text = trimmed[position]
        if len(text) > max_chars:
            trimmed[position] = text[:max_chars].rstrip() + TRUNCATION_MARKER
    return trimmed


def write_sample(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def describe(path: Path, header: list[str], rows: list[list[str]]) -> None:
    header_index = {name: i for i, name in enumerate(header)}
    total = len(rows)
    size_mb = path.stat().st_size / 1024 / 1024

    print(f"\nWrote {total:,} rows ({size_mb:.1f} MB) to {path}")
    print(f"Columns: {len(header)}")

    if "smokes" in header_index:
        smokes = [row[header_index["smokes"]] for row in rows]
        known = [value for value in smokes if value]
        smokers = sum(1 for value in known if value in {"sometimes", "when drinking", "yes", "trying to quit"})
        unknown = len(smokes) - len(known)
        print(f"Unknown 'smokes': {unknown:,} rows ({unknown / total:.1%})")
        print(
            f"Smoker share among known: {smokers / len(known):.1%} ({smokers:,} of {len(known):,})"
            if known
            else "Smoker share: n/a"
        )
    if "sign" in header_index:
        signs = {row[header_index["sign"]].split()[0] for row in rows if row[header_index["sign"]]}
        print(f"Distinct astrological signs: {len(signs)}")
    if "sex" in header_index:
        sexes = Counter(row[header_index["sex"]] or "(blank)" for row in rows)
        print(f"Sex split: {dict(sexes)}")
    if "city" not in header_index and "location" in header_index:
        cities = Counter(row[header_index["location"]].split(",")[0].strip() for row in rows)
        print(f"Top 5 locations: {cities.most_common(5)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the committed demo sample from the full profiles.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See DATA.md for where to obtain the full dataset.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"full dataset (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"sample to write (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--rows", type=int, default=10_000, help="rows to sample (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--max-essay-chars",
        type=int,
        default=400,
        help="truncate each essayN column to this many characters; 0 to keep full text (default: 400)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.exists():
        print(f"error: full dataset not found at {args.input}", file=sys.stderr)
        print("       download profiles.csv and place it there, or pass --input PATH.", file=sys.stderr)
        print("       see DATA.md for instructions.", file=sys.stderr)
        return 1

    header = read_header(args.input)
    missing = [column for column in STRATIFY_ON if column not in header]
    if missing:
        print(f"error: expected column(s) {missing} in {args.input}", file=sys.stderr)
        return 1

    print(f"Reading {args.input} ...")
    counts, total = count_strata(args.input, header)
    print(f"  {total:,} rows across {len(counts)} (sex, smokes) groups")

    target = min(args.rows, total)
    quotas = allocate_quotas(counts, target)

    rng = random.Random(args.seed)
    rows = stratified_sample(args.input, header, quotas, rng)

    essay_positions = [header.index(column) for column in ESSAY_COLUMNS if column in header]
    rows = [truncate_essays(row, essay_positions, args.max_essay_chars) for row in rows]

    write_sample(args.output, header, rows)
    describe(args.output, header, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
