# Cupid 2012 — OkCupid Dating Data Studio

**An interactive Streamlit app for exploring 59,946 real OkCupid profiles from the
2012 San Francisco Bay Area — plus an honest machine-learning study on whether
smoking is predictable from the rest of a dating profile.**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-6.3-3F4F75?logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-2ea44f)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://okcupidsmokingpredictor-bb36gkdr5sbeub82b6lz83.streamlit.app/)

---

## The short version

**The app** turns a 151 MB historical dating dataset into something you can actually
poke at: set your own partner criteria and see your real odds, then find out which
single requirement is doing the most damage to your dating pool.

**The study** answers a narrower question honestly, and mostly in the negative.
Smoking *is* weakly predictable from the rest of a profile — but the headline number
that looks like success is not one, and the notebook is largely about proving that.

| Model (held-out test set, 10,887 profiles) | Accuracy | Precision (smoker) | Recall (smoker) | F1 (smoker) |
|---|---|---|---|---|
| Baseline — always answer "does not smoke" | **0.806** | 0.000 | 0.000 | 0.000 |
| Logistic regression | 0.820 | 0.618 | 0.184 | 0.283 |
| Logistic regression, `class_weight="balanced"` | 0.709 | 0.351 | **0.592** | 0.441 |
| Random forest, `max_depth=5` | 0.751 | 0.392 | 0.520 | **0.447** |

### What that table actually says

**80.6% of the people who answered the smoking question do not smoke.** So the plain
logistic regression's 82.0% accuracy is not a result — it is the base rate plus a
rounding error, and that model catches only 18% of actual smokers. Twelve points of
accuracy came from learning to say "no" more confidently, not from learning anything.

Balancing the class weights fixes the right thing and wrecks the headline number:
accuracy drops to 70.9% while smoker recall climbs from 0.18 to 0.59. If the goal is
*finding* smokers rather than *scoring* well, that trade is worth making — and it is
completely invisible if you only ever report accuracy.

The signal that does exist is almost entirely drug use and drinking (`drugs_often`
leads with a coefficient of 2.07, `drinks_very often` 1.02), while age is close to
irrelevant at -0.04. That is nearly tautological: those are co-reported habits from
the same session, not independent evidence.

The most useful thing the study does is refuse to overclaim. Five-fold
cross-validation shows depths 3, 5 and 7 land at 0.441 / 0.443 / 0.444 mean F1 —
inside one fold-to-fold standard deviation of each other — so "depth 7 is best" is
reading noise. An earlier version of this analysis did exactly that; the notebook
shows the corrected version with the numbers attached.

→ **[Read the full analysis](notebooks/okcupid-smoking-prediction.ipynb)**

---

## Screenshots

### Match Calculator — set your criteria, see the odds

![Match Calculator](assets/match-calculator.png)

Twelve filters, a live probability card, and a **dealbreaker funnel** that shows how
many candidates each requirement removes. The panel on the right names the single
biggest bottleneck — in the default configuration, filtering by gender alone
eliminates 59.8% of the pool before anything else gets a say.

### Bay Area 2012 Data Studio — five tabs of exploration

![Data Studio](assets/data-studio.png)

Age, orientation, height and relationship status; drinking, smoking and drug habits
by age cohort; occupations and self-reported income from the 2012 Silicon Valley
boom; religion, how seriously people take it, and the zodiac; and the geography of
the Bay Area dating pool.

### Profile Detective — read the actual profiles

![Profile Detective](assets/profile-detective.png)

Full-text search across the ten OkCupid essay prompts, then a random matching
profile rendered with everything the dataset records about them. This is the part
that makes the dataset feel like people rather than rows.

---

## Quickstart

**[Try the live demo](https://okcupidsmokingpredictor-bb36gkdr5sbeub82b6lz83.streamlit.app/)** — no install needed.

Or run it locally:

```bash
git clone https://github.com/ShadowstheDark/OK_Cupid_Smoking_predictor.git
cd OK_Cupid_Smoking_predictor
pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501>. On Windows you can also double-click `run_app.bat`.

The repository ships a **10,000-row stratified sample** (`data/sample_profiles.csv`,
~16 MB) so this works immediately. Drop the full 59,946-row `profiles.csv` into
`data/full/` whenever you want the complete dataset — the app detects it and switches
automatically. See **[DATA.md](DATA.md)** for provenance and instructions.

Deep links open the app on a specific mode:

| URL | Opens |
|---|---|
| `/?page=calculator` | Match Calculator |
| `/?page=studio` | Bay Area 2012 Data Studio |
| `/?page=detective` | Profile Detective |

---

## Deploy your own

1. Push this repository to GitHub.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **New app** → pick the repository, branch `main`, main file path `app.py`.
4. **Deploy.** First build takes a couple of minutes.
5. Copy the resulting `https://<name>.streamlit.app` URL into the badge at the top of
   this file.

The sample data is committed specifically so this works with no extra setup — Streamlit
Community Cloud will not have your local `data/full/`.

---

## Project structure

```
.
├── app.py                              # the Streamlit app: 3 modes, custom CSS
├── data_loader.py                      # loading, cleaning and derived columns
├── run_app.bat                         # Windows launcher
├── requirements.txt                    # app dependencies (pinned)
├── requirements-notebook.txt           # + scikit-learn, for the analysis
├── .streamlit/config.toml              # dark theme matching the CSS
├── scripts/
│   └── build_sample.py                 # regenerates the committed sample (stdlib only)
├── data/
│   ├── sample_profiles.csv             # committed, 10,000 rows
│   └── full/                           # git-ignored, drop the full file here
├── notebooks/
│   └── okcupid-smoking-prediction.ipynb
├── assets/                             # screenshots
├── DATA.md                             # provenance, schema, ethics
└── LICENSE                             # MIT — code only, not the data
```

### How the app is built

A dashboard is only as trustworthy as the numbers behind it, so the details that
matter are explicit rather than hidden:

- **One source of truth for derived data.** `data_loader.py` owns every cleaning
  decision and exposes `EDUCATION_CATEGORIES` — the UI builds its filter dropdowns
  from that constant, so the two can never drift apart.
- **No hard-coded totals.** Row counts are read from whatever dataset loaded, so the
  sample and the full extract are both self-consistent.
- **Stratified sampling, on purpose.** The sample is drawn proportionally within each
  `(sex, smokes)` group so the demographics *and* the 19% smoker base rate survive;
  a plain random sample would distort exactly the numbers the notebook depends on.
- **A friendly failure mode.** A missing dataset produces an instruction, not a
  traceback.

---

## Tech

Streamlit · pandas · NumPy · Plotly · scikit-learn · custom CSS

---

## What I'd do next

- **Put the model in the app.** The two halves of this project currently sit next to
  each other rather than talking: the notebook trains a smoking model that the
  interface never surfaces. A fourth mode that scores a chosen profile live would
  join them up.
- **Treat "did not answer" as its own level.** 11,317 rows have no drug answer and
  are currently encoded identically to "never", which is the weakest point in the
  feature set.
- **Use the text.** Ten columns of essays were never touched, and if anything in this
  dataset carries signal beyond the structured fields, it is there.
- **Report precision-recall curves** so the operating point is chosen deliberately,
  rather than by picking between two fitted models after the fact.

---

## Data and ethics

The dataset is Codecademy's OkCupid extract: public profiles from the 2012 Bay Area,
including the free-text essays real people wrote about themselves. It is not mine,
and it is not covered by the MIT licence on this code.

Because that writing is personal, this repository ships a **truncated 10,000-row
sample** and points to the source rather than rehosting the full corpus. Full
provenance, schema and rebuild instructions are in **[DATA.md](DATA.md)**.

## Licence

MIT for the code — see [LICENSE](LICENSE). The dataset is excluded from that licence.
