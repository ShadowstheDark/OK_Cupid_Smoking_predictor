# Data

## What ships with the repository

| Path | Committed? | Rows | Size | Purpose |
|---|---|---|---|---|
| `data/sample_profiles.csv` | yes | 10,000 | ~16 MB | Lets the app run straight after a clone and powers the live demo |
| `data/full/profiles.csv` | **no** (git-ignored) | 59,946 | ~151 MB | The complete 2012 extract, used by the notebook |

The full file is excluded because it is **over GitHub's 100 MB per-file limit** —
a push containing it is rejected outright, not merely discouraged.

## Provenance

The data is an extract of public OkCupid profiles from the greater San Francisco
Bay Area, collected in 2012 and distributed by **Codecademy** as the dataset for
its *Date-A-Scientist* project.

It is **not** owned by, licensed to, or maintained by the author of this
repository, and the MIT licence on the code does not extend to it.

### A note on the content

These are real people. The profiles include the free-text essays users wrote to
describe themselves — hobbies, jobs, families, health, private admissions. That
material was public when it was scraped in 2012, but it was never written with
this kind of reuse in mind, and the people who wrote it did not consent to
appearing in a portfolio project.

That is why the repository ships **10,000 truncated rows rather than the whole
file**, and links to the source instead of rehosting it. Please keep it that way:
run your analysis, publish your results, but do not republish the corpus.

## Schema

31 columns, 59,946 rows.

| Column | Type | Notes |
|---|---|---|
| `age` | numeric | Values outside 18–85 are nulled by the loader |
| `body_type`, `diet`, `drinks`, `drugs`, `education`, `job`, `offspring`, `orientation`, `pets`, `religion`, `sex`, `smokes`, `speaks`, `status`, `ethnicity` | categorical | Lower-cased; blanks become `"unspecified"` |
| `height` | numeric | Inches; values outside 50–86 are nulled |
| `income` | numeric | Mostly undisclosed (`-1`) |
| `last_online` | timestamp | `2012-MM-DD-HH-MM` |
| `location` | text | `"san francisco, california"` |
| `sign` | text | Astrological sign, with qualifiers: `"leo and it&rsquo;s fun to think about"` |
| `essay0` … `essay9` | text | The ten free-text prompts, HTML-escaped |

The loader in `data_loader.py` derives these additional columns: `religion_base`,
`religion_seriousness`, `city`, `edu_category`, `likes_dogs`, `likes_cats`,
`has_dogs`, `has_cats`, `height_cm`, `astrology_sign`.

## Getting the full dataset

1. Obtain `profiles.csv` from the Codecademy *Date-A-Scientist* exercise (or the
   original project distribution you downloaded it from).
2. Save it to `data/full/profiles.csv`.
3. Nothing else is needed — `data_loader.get_csv_path()` prefers the full extract
   over the sample automatically, and the app's sidebar will report the larger row
   count.

You can also point the loader anywhere without moving files:

```bash
# Windows (cmd)
set OKCUPID_CSV=D:\datasets\profiles.csv && streamlit run app.py

# macOS / Linux
OKCUPID_CSV=/path/to/profiles.csv streamlit run app.py
```

## How the sample was built

`scripts/build_sample.py` regenerates `data/sample_profiles.csv`. It uses only the
standard library, so no install is required:

```bash
python scripts/build_sample.py
python scripts/build_sample.py --rows 5000 --seed 7 --max-essay-chars 600
```

Three deliberate choices:

- **Stratified on `(sex, smokes)`**, sampled proportionally within each group. A
  plain random sample of a few thousand rows would distort the 19% smoker base
  rate that the notebook's whole argument rests on, and would thin the demographic
  groups the app charts. The sample keeps the smoker share at 19.4% and all 12
  astrological signs populated.
- **10,000 rows, not 2,000.** The Match Calculator chains twelve filters together,
  so a small sample returns "Zero Matches Found" for perfectly ordinary criteria
  and makes the demo look broken.
- **Essays truncated to 400 characters**, which is the only reason the file is
  16 MB instead of 151 MB byte-for-byte. The ten essay columns are ~85% of the
  bytes. The cap is a compromise: much shorter would shrink the file further but
  would gut the Profile Detective page, which shows real bios and searches them.
  Truncated text is marked with a trailing `…`.

Reproducing the committed sample exactly:

```bash
python scripts/build_sample.py --rows 10000 --seed 42 --max-essay-chars 400
```

## Reproducing the notebook

`notebooks/okcupid-smoking-prediction.ipynb` needs the **full** extract. Install the
extra dependencies and run it from the repository root:

```bash
pip install -r requirements-notebook.txt
jupyter notebook notebooks/okcupid-smoking-prediction.ipynb
```

Its committed outputs were produced on the full 59,946-row file, so the numbers in
the markdown match the numbers in the cells.
