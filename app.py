from typing import Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_loader import (
    EDUCATION_CATEGORIES,
    FULL_DATASET_ROWS,
    clean_html,
    load_data,
    using_sample,
)
from smoking_model import (
    DRUG_OPTIONS,
    DRINK_OPTIONS,
    BODY_TYPE_OPTIONS,
    SMOKER_POSITIVE_LABELS,
    get_model_suite,
    predict_single_profile,
)

# ---------------------------------------------------------
# Page Configuration & Custom Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cupid 2012 | Smoking Predictor & Bay Area Dating Studio",
    page_icon="💘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient Headers */
    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff2a5f 0%, #ff7eb3 50%, #9055ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 42, 95, 0.4);
    }
    .metric-number {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ff2a5f;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #cbd5e1;
        font-weight: 600;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    /* Badge Pills */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-pink { background: rgba(255, 42, 95, 0.15); color: #ff5e87; border: 1px solid rgba(255, 42, 95, 0.3); }
    .badge-purple { background: rgba(144, 85, 255, 0.15); color: #b185ff; border: 1px solid rgba(144, 85, 255, 0.3); }
    .badge-cyan { background: rgba(6, 182, 212, 0.15); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-emerald { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }

    /* Profile Cards */
    .profile-card {
        background: #111420;
        border: 1px solid #23293e;
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .profile-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
    }
    .essay-snippet {
        font-size: 0.92rem;
        line-height: 1.6;
        color: #94a3b8;
        background: rgba(0,0,0,0.25);
        padding: 0.8rem 1rem;
        border-left: 3px solid #ff2a5f;
        border-radius: 0 8px 8px 0;
        margin-top: 0.8rem;
    }

    /* Custom Alert Boxes */
    .callout-box {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        background: rgba(255, 42, 95, 0.08);
        border: 1px solid rgba(255, 42, 95, 0.25);
        margin: 1rem 0;
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Dataset with Cache
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading OkCupid profiles from 2012...")
def get_dataset():
    return load_data()


try:
    df_raw = get_dataset()
except FileNotFoundError as err:
    st.error("**Could not find the dataset.**\n\n" + str(err))
    st.stop()

# Total rows actually loaded. On the full extract this is 59,946; on the
# committed demo sample it is smaller, so every count below is derived from it
# rather than hard-coded.
TOTAL_PROFILES = len(df_raw)
IS_SAMPLE = using_sample()

@st.cache_resource(show_spinner="Training ML models & benchmarking suite...")
def get_cached_model_suite():
    return get_model_suite()


# ---------------------------------------------------------
# Helpers for the criteria funnel
# ---------------------------------------------------------
# Every filter below treats an unanswered question the same way: by default a blank
# answer fails the criterion, so all twelve filters judge candidates by the same rule.
# The toggle in the criteria panel relaxes that, and either way the funnel reports how
# many removals came from missing answers rather than genuine mismatches.


def missing_answer(pool: pd.DataFrame, column: str) -> pd.Series:
    """Rows in `pool` with no usable answer for `column`.

    data_loader writes the literal "unspecified" into the categorical columns and
    leaves age, height and the pet flags as NaN, so both shapes are handled here.
    """
    series = pool[column]
    return series.isna() if series.dtype.kind == "f" else series.eq("unspecified")


def filter_step(
    label: str, pool: pd.DataFrame, keep: pd.Series, blank: Optional[pd.Series] = None
) -> Tuple[Tuple[str, int, int], pd.DataFrame]:
    """One funnel stage: its row and the narrowed pool.

    `blank` marks the candidates whose answer to this criterion was missing, so the
    funnel can attribute attrition to that rather than to a real mismatch.
    """
    dropped_blanks = int((blank & ~keep).sum()) if blank is not None else 0
    return (label, int(keep.sum()), dropped_blanks), pool[keep]


@st.cache_data(show_spinner=False)
def answer_rates(_df: pd.DataFrame, column: str) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Smoker share by the answer given in `column`, among profiles that answered `smokes`.

    The leading underscore keeps the DataFrame out of the cache key, so this is computed
    once per column per process rather than on every widget interaction.
    """
    answered = _df[_df["smokes"].isin(SMOKER_POSITIVE_LABELS | {"no"})]
    is_smoker = answered["smokes"].isin(SMOKER_POSITIVE_LABELS)
    grouped = is_smoker.groupby(answered[column].astype(str))
    return (
        {key: float(value) for key, value in grouped.mean().items()},
        {key: int(value) for key, value in grouped.size().items()},
    )


def model_coefficient(impact_df: pd.DataFrame, feature: str) -> Optional[float]:
    """Balanced logistic-regression coefficient for one feature, or None if absent."""
    row = impact_df[impact_df["feature"] == feature]
    return float(row["lr_balanced_coef"].iloc[0]) if len(row) else None


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
# Each mode is addressable, so a link can open straight into one of them:
#   ?page=calculator   ?page=predictor   ?page=model_studio   ?page=studio   ?page=detective
PAGE_OPTIONS = [
    "Match Calculator (Reality Check)",
    "Smoking Predictor (Live Model)",
    "Model Studio & Code Breakdown",
    "Bay Area 2012 Data Studio",
    "Profile Detective",
]
PAGE_SLUGS = {
    "calculator": 0,
    "predictor": 1,
    "predict": 1,
    "model_studio": 2,
    "model": 2,
    "studio": 3,
    "detective": 4,
}

with st.sidebar:
    st.markdown("## OkCupid 2012 Analytics")
    st.caption(f"San Francisco Bay Area • {TOTAL_PROFILES:,} Historical Profiles")
    if IS_SAMPLE:
        st.caption(
            f"Demo sample of the full {FULL_DATASET_ROWS:,}-profile 2012 extract. "
            "Stratified on sex and smoking status, so percentages stay representative."
        )
    
    requested_page = str(st.query_params.get("page", "")).lower()
    mode = st.radio(
        "Navigation",
        PAGE_OPTIONS,
        index=PAGE_SLUGS.get(requested_page, 0),
    )
    st.divider()
    st.markdown("""
    **About the Dataset:**
    Data collected from public OkCupid profiles active in the greater San Francisco Bay Area during 2012.
    """)
    st.caption("SF Bay Area Singles Pool: ~1.2M adults (2012 US Census est.)")

# ---------------------------------------------------------
# TAB 1: REALITY CHECK (MATCH CALCULATOR)
# ---------------------------------------------------------
if mode == "Match Calculator (Reality Check)":
    st.markdown('<div class="hero-title">2012 Reality Check: California Match Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Set your partner criteria to calculate your exact statistical odds of finding a match in the 2012 San Francisco Bay Area.</div>', unsafe_allow_html=True)

    # Filter Controls Grid
    with st.expander("Dating Criteria & Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Basics")
            pref_gender = st.selectbox(
                "I am seeking:",
                ["Women", "Men", "Any Gender"],
                index=0,
                help="Filter by gender recorded on profiles"
            )
            
            age_min, age_max = st.slider(
                "Age Range:",
                min_value=18,
                max_value=70,
                value=(22, 35),
                step=1
            )
            
            pref_orientation = st.multiselect(
                "Sexual Orientation:",
                ["straight", "gay", "bisexual"],
                default=["straight", "bisexual"],
                help="Orientations matching your dating pool"
            )

            pref_status = st.multiselect(
                "Relationship Status:",
                ["single", "available", "seeing someone", "married"],
                default=["single", "available"]
            )

        with col2:
            st.markdown("##### Habits & Lifestyle")
            drink_options = ["socially", "rarely", "often", "not at all", "very often", "desperately"]
            pref_drinks = st.multiselect(
                "Drinking Habits:",
                options=drink_options,
                default=["socially", "rarely", "not at all"],
                help="Leave all or select specific habits"
            )
            
            smoke_options = ["no", "sometimes", "when drinking", "yes", "trying to quit"]
            pref_smokes = st.multiselect(
                "Smoking Habits:",
                options=smoke_options,
                default=["no", "sometimes", "when drinking"],
                help="Cigarette smoking preference"
            )

            drug_options = ["never", "sometimes", "often"]
            pref_drugs = st.multiselect(
                "Drug Usage:",
                options=drug_options,
                default=["never", "sometimes"],
                help="Recreational drug usage"
            )

            pref_pets = st.selectbox(
                "Pet Compatibility:",
                ["No Preference", "Must Like/Have Dogs", "Must Like/Have Cats", "Must Like Both Dogs & Cats"],
                index=0
            )

        with col3:
            st.markdown("##### Beliefs & Demographics")
            
            rel_options = ["agnosticism", "atheism", "christianity", "catholicism", "judaism", "buddhism", "hinduism", "islam", "other"]
            pref_rel = st.multiselect(
                "Religion / Worldview:",
                options=rel_options,
                default=[],
                help="Select one or more faiths. Leave blank for any."
            )

            pref_seriousness = st.multiselect(
                "Attitude towards Religion:",
                options=["laughing about it", "not too serious", "somewhat serious", "very serious", "matter-of-fact"],
                default=[],
                help="Optional: filter how seriously they take religion. Leave blank for any."
            )

            # Sourced from data_loader so the filter options can never drift out
            # of sync with the categories the loader actually produces.
            pref_edu = st.multiselect(
                "Education Level:",
                options=EDUCATION_CATEGORIES,
                default=[],
                help="Leave empty for any education level"
            )

            height_min, height_max = st.slider(
                "Height (Inches):",
                min_value=55,
                max_value=82,
                value=(58, 76),
                format="%d in"
            )
            h_min_ft = f"{height_min // 12}'{height_min % 12}\""
            h_max_ft = f"{height_max // 12}'{height_max % 12}\""
            st.caption(f"Converted: **{h_min_ft}** ({round(height_min*2.54)}cm) to **{h_max_ft}** ({round(height_max*2.54)}cm)")

            top_cities = ["All Bay Area", "san francisco", "oakland", "berkeley", "san mateo", "palo alto", "san jose", "alameda"]
            pref_city = st.selectbox("Location:", top_cities, index=0)

            st.markdown("---")
            pref_count_blanks = st.checkbox(
                "Count profiles that left a question blank",
                value=False,
                help=(
                    "Off (default): an unanswered question fails that criterion, so every "
                    "filter judges candidates by the same rule and the funnel compares like "
                    "with like. On: profiles that skipped a question are kept on the grounds "
                    "that we cannot tell, which is more forgiving but quietly mixes answers "
                    "with non-answers."
                ),
            )

    # ---------------------------------------------------------
    # Calculation & Funnel Simulation
    # ---------------------------------------------------------
    current_df = df_raw.copy()
    funnel_steps = [("All 2012 Profiles", len(current_df), 0)]

    def narrow(label, keep, blank):
        """Apply one criterion to the pool, recording the stage it produced."""
        step, narrowed = filter_step(label, current_df, keep, blank)
        funnel_steps.append(step)
        return narrowed

    def keep_with_blanks(keep, blank):
        """The toggle decides whether an unanswered question waives the criterion."""
        return keep | blank if pref_count_blanks else keep

    # 1. Gender Filter
    if pref_gender != "Any Gender":
        keep = current_df['sex'].eq('f' if pref_gender == "Women" else 'm')
        blank = missing_answer(current_df, 'sex')
        current_df = narrow("Gender Filter", keep_with_blanks(keep, blank), blank)

    # 2. Orientation
    if pref_orientation:
        keep = current_df['orientation'].isin(pref_orientation)
        blank = missing_answer(current_df, 'orientation')
        current_df = narrow("Orientation", keep_with_blanks(keep, blank), blank)

    # 3. Age Range
    keep = current_df['age'].between(age_min, age_max)
    blank = missing_answer(current_df, 'age')
    current_df = narrow("Age Window", keep_with_blanks(keep, blank), blank)

    # 4. Status
    if pref_status:
        keep = current_df['status'].isin(pref_status)
        blank = missing_answer(current_df, 'status')
        current_df = narrow("Status", keep_with_blanks(keep, blank), blank)

    # 5. Height
    keep = current_df['height'].between(height_min, height_max)
    blank = missing_answer(current_df, 'height')
    current_df = narrow("Height Range", keep_with_blanks(keep, blank), blank)

    # 6. Smoking
    if pref_smokes:
        keep = current_df['smokes'].isin(pref_smokes)
        blank = missing_answer(current_df, 'smokes')
        current_df = narrow("Smoking Habits", keep_with_blanks(keep, blank), blank)

    # 7. Drinking
    if pref_drinks:
        keep = current_df['drinks'].isin(pref_drinks)
        blank = missing_answer(current_df, 'drinks')
        current_df = narrow("Drinking Habits", keep_with_blanks(keep, blank), blank)

    # 8. Drugs
    if pref_drugs:
        keep = current_df['drugs'].isin(pref_drugs)
        blank = missing_answer(current_df, 'drugs')
        current_df = narrow("Drug Habits", keep_with_blanks(keep, blank), blank)

    # 9. Religion — belief and how seriously it is taken count as one stage
    if pref_rel or pref_seriousness:
        keep = pd.Series(True, index=current_df.index)
        blank = pd.Series(False, index=current_df.index)
        if pref_rel:
            keep &= current_df['religion_base'].isin(pref_rel)
            blank |= missing_answer(current_df, 'religion_base')
        if pref_seriousness:
            keep &= current_df['religion_seriousness'].isin(pref_seriousness)
            blank |= missing_answer(current_df, 'religion_seriousness')
        current_df = narrow("Religion / Beliefs", keep_with_blanks(keep, blank), blank)

    # 10. Education
    if pref_edu:
        keep = current_df['edu_category'].isin(pref_edu)
        blank = missing_answer(current_df, 'edu_category')
        current_df = narrow("Education", keep_with_blanks(keep, blank), blank)

    # 11. Pets
    if pref_pets != "No Preference":
        pet_columns = {
            "Must Like/Have Dogs": ['likes_dogs'],
            "Must Like/Have Cats": ['likes_cats'],
            "Must Like Both Dogs & Cats": ['likes_dogs', 'likes_cats'],
        }[pref_pets]
        keep = current_df[pet_columns[0]].eq(True)
        blank = missing_answer(current_df, pet_columns[0])
        for column in pet_columns[1:]:
            keep &= current_df[column].eq(True)
            blank |= missing_answer(current_df, column)
        current_df = narrow("Pet Preference", keep_with_blanks(keep, blank), blank)

    # 12. Location
    if pref_city != "All Bay Area":
        keep = current_df['city'].eq(pref_city)
        blank = missing_answer(current_df, 'city')
        current_df = narrow("Location", keep_with_blanks(keep, blank), blank)

    # Compute Statistics
    final_matches = len(current_df)
    match_pct = (final_matches / TOTAL_PROFILES) * 100
    odds_ratio = int(round(TOTAL_PROFILES / final_matches)) if final_matches > 0 else "∞"
    
    # Estimate real-world 2012 Bay Area singles pool (~1,200,000 active singles)
    real_world_pool = 1200000
    estimated_real_singles = int(round(real_world_pool * (final_matches / TOTAL_PROFILES)))

    # Reality Rating Classification
    if final_matches == 0:
        rating_badge = "Zero Matches Found"
        rating_class = "badge-pink"
        rating_desc = f"No profile in all {TOTAL_PROFILES:,} users matches every one of your criteria. Consider relaxing height, religion, or habit filters."
    elif match_pct < 0.1:
        rating_badge = "Extremely Rare (<0.1%)"
        rating_class = "badge-pink"
        rating_desc = "Less than 1 in 1,000 users in 2012 California meet these specific standards."
    elif match_pct < 1.0:
        rating_badge = "High Standards (0.1% - 1.0%)"
        rating_class = "badge-amber"
        rating_desc = "Selective standards. Matches exist, but require looking through a substantial pool of candidates."
    elif match_pct < 5.0:
        rating_badge = "Selective Match (1.0% - 5.0%)"
        rating_class = "badge-purple"
        rating_desc = "Balanced taste with clear preferences. Thousands of potential matches across the Bay Area."
    elif match_pct < 20.0:
        rating_badge = "Realistic Match (5.0% - 20.0%)"
        rating_class = "badge-cyan"
        rating_desc = "Well-grounded criteria with a healthy, diverse dating pool."
    else:
        rating_badge = "Broad Dating Pool (>20.0%)"
        rating_class = "badge-emerald"
        rating_desc = "A large portion of the 2012 California singles population fits this profile."

    st.markdown("---")

    # Big Impact Metric Cards
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{match_pct:.2f}%</div>
            <div class="metric-label">Match Probability</div>
            <div class="metric-sub">of 2012 OkCupid users</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol2:
        odds_display = f"1 in {odds_ratio:,}" if odds_ratio != "∞" else "None"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number" style="color: #9055ff;">{odds_display}</div>
            <div class="metric-label">Odds of A Match</div>
            <div class="metric-sub">random encounters</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number" style="color: #06b6d4;">{final_matches:,}</div>
            <div class="metric-label">Matching Profiles</div>
            <div class="metric-sub">out of {TOTAL_PROFILES:,} in dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number" style="color: #10b981;">~{estimated_real_singles:,}</div>
            <div class="metric-label">Est. Bay Area Singles</div>
            <div class="metric-sub">2012 census projection</div>
        </div>
        """, unsafe_allow_html=True)

    # Rating Callout
    st.markdown(f"""
    <div class="callout-box">
        <span class="badge {rating_class}" style="font-size: 0.95rem; padding: 0.35rem 0.9rem;">{rating_badge}</span>
        <div style="margin-top: 0.5rem; font-size: 1rem;">{rating_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    # Calculation Methodology Expander
    with st.expander("ℹ️ How are these statistics calculated?"):
        st.markdown(f"""
        - **Match Probability (%)**: Calculated as `(Matching Profiles / Total {TOTAL_PROFILES:,} Profiles) × 100`.
        - **Odds of a Match ("1 in X")**: The reciprocal of the match probability (`Total Profiles / Matching Profiles`). For instance, if 0.5% of profiles match, the odds are `1 in 200`.
        - **Estimated Bay Area Singles**: According to the 2012 U.S. Census Bureau American Community Survey, the 9-county San Francisco Bay Area had roughly **{real_world_pool:,} unmarried adult residents**. Multiplying this population by your match probability gives a realistic real-world estimate of individuals fitting your criteria.
        - **Dealbreaker Funnel**: Sequentially applies each criterion and measures how many candidates remain at each stage, as well as the marginal percentage dropped. Every criterion applies the same rule to an unanswered question — by default it fails the criterion — and each step also reports how much of its attrition came from blank answers rather than real mismatches.
        """)

    # ---------------------------------------------------------
    # Visual Dealbreaker Funnel
    # ---------------------------------------------------------
    f_col1, f_col2 = st.columns([3, 2])

    with f_col1:
        st.markdown("#### The Dealbreaker Funnel")
        st.caption("Attrition across consecutive filters:")

        funnel_df = pd.DataFrame(funnel_steps, columns=["Step", "Remaining", "Blanks"])

        # Calculate percentage drop at each step
        funnel_df["Drop"] = funnel_df["Remaining"].shift(1) - funnel_df["Remaining"]
        funnel_df["Drop_Pct"] = (funnel_df["Drop"] / funnel_df["Remaining"].shift(1)) * 100
        funnel_df["Drop_Pct"] = funnel_df["Drop_Pct"].fillna(0)

        total_blanks = int(funnel_df["Blanks"].sum())

        fig_funnel = go.Figure(go.Funnel(
            y=funnel_df["Step"],
            x=funnel_df["Remaining"],
            textinfo="value+percent previous",
            marker={
                "color": ["#ff2a5f", "#ff4b7a", "#ff6b95", "#9055ff", "#a370ff", "#06b6d4", "#22d3ee", "#38bdf8", "#10b981", "#34d399", "#fbbf24", "#f59e0b", "#e11d48"],
                "line": {"width": 1, "color": "#1e293b"}
            }
        ))
        fig_funnel.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
            height=420
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

        if pref_count_blanks:
            st.caption(
                "Every criterion is judged the same way: a candidate who left a question "
                "blank is kept rather than dropped."
            )
        elif total_blanks:
            st.caption(
                f"{total_blanks:,} candidates were removed because they never answered the "
                "question behind a filter, not because they gave the wrong answer. Tick "
                "*\"Count profiles that left a question blank\"* to keep them."
            )

    with f_col2:
        st.markdown("#### Dealbreaker Breakdown")
        st.caption("Largest single factor reducing candidates:")

        def attribution(row) -> str:
            """Say how much of a removal was unanswered questions rather than mismatches."""
            blanks = int(row['Blanks'])
            if blanks == 0:
                return ""
            return f", {blanks:,} of them unanswered"

        drops = funnel_df[funnel_df["Drop"] > 0].sort_values(by="Drop_Pct", ascending=False)
        if not drops.empty:
            biggest = drops.iloc[0]
            st.warning(f"**Biggest Bottleneck: {biggest['Step']}**\n\nEliminated **{biggest['Drop_Pct']:.1f}%** ({int(biggest['Drop']):,} candidates) of the remaining pool{attribution(biggest)}.")
            
            st.markdown("##### Other Major Reductions:")
            for idx, row in drops.iloc[1:5].iterrows():
                st.markdown(f"- **{row['Step']}**: eliminated **{row['Drop_Pct']:.1f}%** ({int(row['Drop']):,} candidates){attribution(row)}")
        else:
            st.info("No single filter eliminated candidates; your criteria match everyone in this pool.")

    # ---------------------------------------------------------
    # Sample Real Matching Profiles
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("#### Sample Matching Profiles")
    st.caption("Anonymous OkCupid profiles matching your exact criteria from 2012:")

    if final_matches > 0:
        sample_size = min(3, final_matches)
        
        # Add a refresh button with session state
        if "sample_seed" not in st.session_state:
            st.session_state.sample_seed = 42
        
        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            if st.button("Shuffle Matches"):
                st.session_state.sample_seed = np.random.randint(1, 100000)
                st.rerun()

        samples = current_df.sample(n=sample_size, random_state=st.session_state.sample_seed)
        
        cols = st.columns(sample_size)
        for i, (_, row) in enumerate(samples.iterrows()):
            with cols[i]:
                age = int(row['age']) if not pd.isna(row['age']) else "?"
                sex = "Female" if row['sex'] == 'f' else "Male"
                height_str = f"{int(row['height'] // 12)}'{int(row['height'] % 12)}\"" if not pd.isna(row['height']) else "?"
                job = row['job'].title() if row['job'] != 'unspecified' else "Unspecified Career"
                city = row['city'].title()
                religion_str = row['religion_base'].title() if row['religion_base'] != 'unspecified' else "Not specified"
                drink_str = row['drinks'].capitalize()
                smoke_str = row['smokes'].capitalize()
                sign = row['astrology_sign'].capitalize()

                # Clean self summary snippet
                raw_bio = row.get('essay0', '')
                cleaned_bio = clean_html(raw_bio)
                if not cleaned_bio or len(cleaned_bio) < 20:
                    cleaned_bio = clean_html(row.get('essay1', 'No bio provided.'))
                if len(cleaned_bio) > 280:
                    cleaned_bio = cleaned_bio[:280] + "..."

                st.markdown(f"""
                <div class="profile-card">
                    <div class="profile-header">{age} • {sex} • {city}</div>
                    <div>
                        <span class="badge badge-pink">{height_str}</span>
                        <span class="badge badge-purple">{job}</span>
                        <span class="badge badge-cyan">{religion_str}</span>
                        <span class="badge badge-amber">{sign}</span>
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #94a3b8;">
                        Drinks: {drink_str} • Smokes: {smoke_str}
                    </div>
                    <div class="essay-snippet">
                        <em>"{cleaned_bio}"</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No sample profiles found for this combination of filters. Loosen some restrictions above.")

# ---------------------------------------------------------
# TAB 2: SMOKING PREDICTOR (LIVE MODEL)
# ---------------------------------------------------------
elif mode == "Smoking Predictor (Live Model)":
    st.markdown('<div class="hero-title">Smoking Predictor: Live ML Inference</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Real-time prediction of smoking behavior from demographic and lifestyle factors, powered by class-balanced machine learning.</div>',
        unsafe_allow_html=True,
    )

    suite = get_cached_model_suite()

    # Preset Profiles Quick-Loader
    st.markdown("##### ⚡ Quick-Load Persona Presets")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        if st.button("🏃 Clean Living Athlete", use_container_width=True):
            st.session_state["pred_age"] = 28
            st.session_state["pred_drugs"] = "never"
            st.session_state["pred_drinks"] = "not at all"
            st.session_state["pred_body"] = "athletic"
            st.session_state["actual_smoke_label"] = None
            st.rerun()
    with col_p2:
        if st.button("🍸 Weekend Socialite", use_container_width=True):
            st.session_state["pred_age"] = 26
            st.session_state["pred_drugs"] = "never"
            st.session_state["pred_drinks"] = "socially"
            st.session_state["pred_body"] = "fit"
            st.session_state["actual_smoke_label"] = None
            st.rerun()
    with col_p3:
        if st.button("🎉 Party Regular", use_container_width=True):
            st.session_state["pred_age"] = 23
            st.session_state["pred_drugs"] = "sometimes"
            st.session_state["pred_drinks"] = "often"
            st.session_state["pred_body"] = "thin"
            st.session_state["actual_smoke_label"] = None
            st.rerun()
    with col_p4:
        if st.button("🎲 Random 2012 Profile", use_container_width=True):
            known_profiles = df_raw[df_raw["smokes"].isin(["no", "sometimes", "when drinking", "yes", "trying to quit"])]
            sample_row = known_profiles.sample(n=1).iloc[0]
            st.session_state["pred_age"] = int(sample_row["age"]) if not pd.isna(sample_row["age"]) else 30
            st.session_state["pred_drugs"] = sample_row["drugs"] if sample_row["drugs"] in DRUG_OPTIONS else "unspecified"
            st.session_state["pred_drinks"] = sample_row["drinks"] if sample_row["drinks"] in DRINK_OPTIONS else "unspecified"
            st.session_state["pred_body"] = sample_row["body_type"] if sample_row["body_type"] in BODY_TYPE_OPTIONS else "unspecified"
            st.session_state["actual_smoke_label"] = sample_row["smokes"]
            st.session_state["actual_bio"] = clean_html(sample_row.get("essay0", ""))
            st.session_state["actual_gender"] = sample_row.get("sex", "unknown")
            st.rerun()

    # Form Controls
    with st.expander("Candidate Profile & Model Controls", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)

        default_age = st.session_state.get("pred_age", 27)
        default_drugs = st.session_state.get("pred_drugs", "never")
        default_drinks = st.session_state.get("pred_drinks", "socially")
        default_body = st.session_state.get("pred_body", "average")

        with fcol1:
            st.markdown("##### Basic Traits")
            input_age = st.slider("Age:", min_value=18, max_value=75, value=default_age, step=1)
            drugs_idx = DRUG_OPTIONS.index(default_drugs) if default_drugs in DRUG_OPTIONS else 0
            input_drugs = st.selectbox(
                "Drug Habits:",
                DRUG_OPTIONS,
                index=drugs_idx,
                help="Recreational drug usage. 'unspecified' fixes the bug where missing was conflated with 'never'."
            )

        with fcol2:
            st.markdown("##### Lifestyle & Body")
            drinks_idx = DRINK_OPTIONS.index(default_drinks) if default_drinks in DRINK_OPTIONS else 0
            input_drinks = st.selectbox("Drinking Frequency:", DRINK_OPTIONS, index=drinks_idx)

            body_idx = BODY_TYPE_OPTIONS.index(default_body) if default_body in BODY_TYPE_OPTIONS else 0
            input_body = st.selectbox("Body Type:", BODY_TYPE_OPTIONS, index=body_idx)

        with fcol3:
            st.markdown("##### Inference Configuration")
            model_choices = [
                "Logistic Regression (Balanced)",
                "HistGradientBoosting (Balanced)",
                "Random Forest (Balanced, max_depth=5)",
                "Logistic Regression (Unweighted)",
            ]
            selected_model_name = st.selectbox(
                "Inference Engine:",
                model_choices,
                index=0,
                help="Select between balanced models (which fix the 80.6% base-rate trap) and the unweighted model."
            )
            input_threshold = st.slider(
                "Decision Threshold:",
                min_value=0.10,
                max_value=0.90,
                value=0.50,
                step=0.05,
                help="Probability cutoff for positive classification. Lowering the threshold catches more smokers (higher recall) at the cost of more false positives."
            )

    profile_dict = {
        "age": input_age,
        "drugs": input_drugs,
        "drinks": input_drinks,
        "body_type": input_body,
    }

    result = predict_single_profile(
        profile_dict,
        model_name=selected_model_name,
        threshold=input_threshold,
        suite=suite,
    )

    # Display Prediction Dashboard
    st.markdown("### Prediction Results & Interpretability")
    res_col1, res_col2 = st.columns([5, 7])

    with res_col1:
        is_smoker = result["prediction"] == 1
        badge_cls = "badge-pink" if is_smoker else "badge-emerald"
        status_text = "Smoker / Social Smoker" if is_smoker else "Non-Smoker"

        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {result['risk_color']}; text-align: left; padding: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="badge {badge_cls}" style="font-size: 0.9rem; padding: 0.4rem 0.9rem;">
                    {status_text}
                </span>
                <span style="font-size: 0.85rem; font-weight: 600; color: {result['risk_color']};">
                    ● {result['risk_level']}
                </span>
            </div>
            <div style="margin-top: 1rem;">
                <div style="font-size: 0.9rem; color: #94a3b8;">Predicted Smoker Probability</div>
                <div class="metric-number" style="color: {result['risk_color']}; font-size: 3.2rem;">
                    {result['probability_percent']}%
                </div>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #cbd5e1;">
                Odds relative to 2012 Bay Area population: 
                <strong style="color: #f1f5f9;">{result['relative_odds_vs_population']}x</strong>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.82rem; color: #64748b;">
                Decision Threshold: {input_threshold:.2f} (Model: {selected_model_name})
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Plotly Risk Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["probability_percent"],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Smoking Risk Meter (%)", 'font': {'size': 16, 'color': '#cbd5e1'}},
            number={'suffix': "%", 'font': {'color': result['risk_color'], 'size': 28}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': result['risk_color']},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 1,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [30, 60], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [60, 100], 'color': "rgba(255, 42, 95, 0.2)"},
                ],
                'threshold': {
                    'line': {'color': "#ffffff", 'width': 3},
                    'thickness': 0.75,
                    'value': input_threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#f1f5f9"},
            height=240,
            margin=dict(l=20, r=20, t=40, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Ground truth validation if random profile was loaded
        actual_label = st.session_state.get("actual_smoke_label")
        if actual_label:
            actual_is_smoker = actual_label in {"sometimes", "when drinking", "yes", "trying to quit"}
            matched = (is_smoker == actual_is_smoker)
            match_color = "#34d399" if matched else "#f87171"
            match_str = "MATCH (Correct Prediction)" if matched else "MISMATCH (Model Error)"
            actual_bio = st.session_state.get("actual_bio", "")
            st.markdown(f"""
            <div class="callout-box" style="border-color: {match_color}; background: rgba(0,0,0,0.3);">
                <div style="font-weight: 700; color: {match_color};">
                    🎯 2012 Profile Ground Truth: {match_str}
                </div>
                <div style="margin-top: 0.4rem; font-size: 0.9rem;">
                    Actual self-reported smoking status: <strong>"{actual_label}"</strong>
                </div>
                {f'<div class="essay-snippet" style="margin-top: 0.6rem; font-size: 0.85rem;"><em>"{actual_bio[:280]}..."</em></div>' if actual_bio else ''}
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("##### 🔍 Feature Attribution: What Drove This Prediction?")
        st.caption("Log-odds contribution of candidate traits relative to reference baselines.")

        attributions = result.get("feature_attributions", [])
        if attributions:
            attr_df = pd.DataFrame(attributions)
            # Exclude base intercept for readable feature comparison
            attr_features = attr_df[attr_df["feature"] != "Baseline (Population Intercept)"].copy()

            if len(attr_features) > 0:
                attr_features["color"] = attr_features["log_odds_impact"].apply(
                    lambda x: "#ff2a5f" if x > 0 else "#10b981"
                )
                attr_features = attr_features.sort_values(by="log_odds_impact", ascending=True)

                fig_attr = px.bar(
                    attr_features,
                    x="log_odds_impact",
                    y="feature",
                    orientation="h",
                    text="log_odds_impact",
                    labels={"log_odds_impact": "Log-Odds Impact (β · x)", "feature": "Profile Trait"},
                    color="log_odds_impact",
                    color_continuous_scale=[[0, "#10b981"], [0.5, "#94a3b8"], [1, "#ff2a5f"]],
                )
                fig_attr.update_traces(
                    texttemplate='%{text:+.2f}',
                    textposition='outside',
                    marker_line_color='rgba(255,255,255,0.2)',
                    marker_line_width=1
                )
                fig_attr.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0.1)",
                    font=dict(color="#cbd5e1"),
                    height=320,
                    margin=dict(l=10, r=20, t=10, b=10),
                    coloraxis_showscale=False,
                    xaxis=dict(gridcolor="#23293e", zerolinecolor="#475569", zerolinewidth=1.5),
                    yaxis=dict(gridcolor="#23293e"),
                )
                st.plotly_chart(fig_attr, use_container_width=True)
            else:
                st.info("All selected traits match the baseline reference categories (never drugs, socially drinks, average body).")
        else:
            st.info(f"Feature attributions are generated for linear models. Currently running {selected_model_name}.")

        # These numbers come from the model that was actually trained for this run, not
        # from a write-up, so they stay true on either dataset.
        drug_rates, drug_counts = answer_rates(df_raw, "drugs")
        impact_df = suite["feature_impact_df"]
        sometimes_coef = model_coefficient(impact_df, "drugs_sometimes")
        blank_coef = model_coefficient(impact_df, "drugs_unspecified")
        age_coef = model_coefficient(impact_df, "age")
        never_rate = drug_rates.get("never", float("nan"))
        blank_rate = drug_rates.get("unspecified", float("nan"))
        blank_profiles = drug_counts.get("unspecified", 0)

        insight_items = []
        if sometimes_coef is not None:
            insight_items.append(
                "<li><strong>Recreational drugs:</strong> the strongest positive coefficient in "
                "the balanced model. Reporting occasional use (<code>drugs_sometimes</code>) "
                f"multiplies the odds of being labelled a smoker by about "
                f"{np.exp(sometimes_coef):.1f}\u00d7 (log-odds {sometimes_coef:+.2f}).</li>"
            )
        if blank_coef is not None:
            insight_items.append(
                "<li><strong>An unanswered question is signal, not noise:</strong> leaving the "
                f"drugs question blank sits at {blank_coef:+.2f} log-odds. Profiles that "
                f"skipped it smoke at {blank_rate:.1%}, against {never_rate:.1%} for those who "
                f"answered <code>never</code> ({blank_profiles:,} profiles in this data).</li>"
            )
        if age_coef is not None:
            insight_items.append(
                "<li><strong>Age:</strong> "
                f"{age_coef:+.3f} log-odds per year \u2014 small enough that age is doing almost "
                "no work here, which is worth knowing before leaning on the model's age "
                "behaviour.</li>"
            )

        st.markdown(
            """
        <div class="callout-box" style="margin-top: 1rem;">
            <strong>💡 What is actually driving these predictions:</strong>
            <ul style="margin-top: 0.4rem; margin-bottom: 0.2rem; padding-left: 1.2rem; font-size: 0.88rem; line-height: 1.5;">
            """
            + "".join(insight_items)
            + """
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# TAB 3: MODEL STUDIO & CODE BREAKDOWN
# ---------------------------------------------------------
elif mode == "Model Studio & Code Breakdown":
    st.markdown('<div class="hero-title">Model Studio & Technical Breakdown</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">The 80.6% accuracy trap, class imbalance mitigation, confusion matrices, precision-recall trade-offs, and production ML architecture.</div>',
        unsafe_allow_html=True,
    )

    suite = get_cached_model_suite()
    metrics = suite["metrics"]
    base_rate = float(suite["base_rate"])
    plain = metrics["Logistic Regression (Unweighted)"]
    balanced = metrics["Logistic Regression (Balanced)"]
    # Kept in a variable: an f-string expression reusing the delimiter's quote type is
    # only legal from Python 3.12, and this file runs on the Cloud's Python too.
    baseline = metrics["Baseline (Always 'No')"]
    test_rows = int(suite["X_test_shape"][0])
    train_rows = int(suite["X_train_shape"][0])
    misses_share = 1 - plain["recall"]

    # 1. The accuracy trap
    st.markdown(f"### 1. The {1 - base_rate:.1%} Accuracy Trap")
    st.caption("Why headline accuracy is a dangerous metric on imbalanced real-world data.")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{1 - base_rate:.1%}</div>
            <div class="metric-label">Non-Smoker Base Rate</div>
            <div class="metric-sub">Majority Class (Class 0)</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{base_rate:.1%}</div>
            <div class="metric-label">Smoker Base Rate</div>
            <div class="metric-sub">Minority Class (Class 1)</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number" style="color: #94a3b8;">{baseline['accuracy']:.1%}</div>
            <div class="metric-label">Baseline Accuracy</div>
            <div class="metric-sub">Scores this while learning nothing at all</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number" style="color: #34d399;">{balanced['recall']:.1%}</div>
            <div class="metric-label">Balanced Model Recall</div>
            <div class="metric-sub">+{(balanced['recall'] - plain['recall']) * 100:.0f} points over the unweighted model</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="callout-box" style="margin-top: 1.2rem;">
        <strong>⚠️ The Accuracy Trap Explained:</strong> Because {1 - base_rate:.1%} of people in this dataset do not smoke, a dummy model that predicts <em>"does not smoke"</em> for every single person already scores <strong>{baseline['accuracy']:.1%}</strong> without learning anything.
        An unweighted Logistic Regression scores <strong>{plain['accuracy']:.1%} accuracy</strong>, which looks like an achievement on paper — but it catches only <strong>{plain['recall']:.1%} of actual smokers</strong>, missing {misses_share:.0%} of them.
        <br><br>
        Re-fitting with <code>class_weight="balanced"</code> re-weights the classes by roughly {1 / (2 * base_rate):.2f} for smokers against {1 / (2 * (1 - base_rate)):.2f} for non-smokers, so a missed smoker costs several times a false alarm. <strong>Accuracy settles at {balanced['accuracy']:.1%} while recall climbs to {balanced['recall']:.1%}</strong> — the headline number gets worse and the model gets more useful.
    </div>
    """, unsafe_allow_html=True)

    # 2. Benchmark Comparison Table
    st.markdown("### 2. Model Benchmark Comparison")
    st.caption(
        f"All five models are trained on the same {train_rows:,} profiles and evaluated on "
        f"the same stratified 20% held-out split ({test_rows:,} profiles), on whichever "
        f"dataset this instance loaded."
    )

    benchmark_rows = []
    for m_name, m_vals in metrics.items():
        benchmark_rows.append({
            "Model Name": m_name,
            "Accuracy": f"{m_vals['accuracy']:.3f}",
            "Precision (Smoker)": f"{m_vals['precision']:.3f}",
            "Recall (Smoker)": f"{m_vals['recall']:.3f}",
            "F1-Score": f"{m_vals['f1']:.3f}",
            "ROC-AUC": f"{m_vals['roc_auc']:.3f}",
        })
    df_benchmark = pd.DataFrame(benchmark_rows)
    st.dataframe(df_benchmark, use_container_width=True, hide_index=True)

    # 3. Interactive Confusion Matrix & Precision-Recall Dynamics
    st.markdown("### 3. Confusion Matrix & Threshold Sensitivity")
    cm_col1, cm_col2 = st.columns([6, 6])

    with cm_col1:
        st.markdown("##### 🔲 Interactive Confusion Matrix")
        chosen_cm_model = st.selectbox(
            "Select Model:",
            list(metrics.keys()),
            index=2, # Logistic Regression (Balanced)
            key="cm_model_select"
        )
        cm_mode = st.radio("Display Format:", ["Normalized Percentages (%)", "Raw Profile Counts"], horizontal=True)

        raw_cm = metrics[chosen_cm_model]["confusion_matrix"]
        norm_cm = metrics[chosen_cm_model]["confusion_matrix_norm"]

        if cm_mode == "Normalized Percentages (%)":
            z_vals = [[val * 100 for val in row] for row in norm_cm]
            text_vals = [
                [f"TN: {z_vals[0][0]:.1f}%", f"FP: {z_vals[0][1]:.1f}%"],
                [f"FN: {z_vals[1][0]:.1f}%", f"TP: {z_vals[1][1]:.1f}%"]
            ]
        else:
            z_vals = raw_cm
            text_vals = [
                [f"TN: {raw_cm[0][0]:,}", f"FP: {raw_cm[0][1]:,}"],
                [f"FN: {raw_cm[1][0]:,}", f"TP: {raw_cm[1][1]:,}"]
            ]

        fig_cm = go.Figure(data=go.Heatmap(
            z=z_vals,
            x=["Predicted Non-Smoker", "Predicted Smoker"],
            y=["Actual Non-Smoker", "Actual Smoker"],
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            colorscale=[[0, "#111420"], [0.5, "#4c1d95"], [1, "#ff2a5f"]],
            showscale=False
        ))
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with cm_col2:
        st.markdown("##### 📈 Decision Threshold Trade-Offs")
        st.caption("How shifting the decision threshold trades off Precision vs Recall.")

        thresh_df = suite["threshold_curve"]
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=thresh_df["threshold"], y=thresh_df["recall"], mode="lines+markers", name="Recall (Smoker)", line=dict(color="#38bdf8", width=3)))
        fig_pr.add_trace(go.Scatter(x=thresh_df["threshold"], y=thresh_df["precision"], mode="lines+markers", name="Precision (Smoker)", line=dict(color="#ff2a5f", width=3)))
        fig_pr.add_trace(go.Scatter(x=thresh_df["threshold"], y=thresh_df["f1"], mode="lines", name="F1-Score", line=dict(color="#fbbf24", width=2, dash="dot")))
        fig_pr.add_trace(go.Scatter(x=thresh_df["threshold"], y=thresh_df["accuracy"], mode="lines", name="Accuracy", line=dict(color="#94a3b8", width=2, dash="dash")))

        fig_pr.add_vline(x=0.50, line_dash="dash", line_color="#34d399", annotation_text="Default 0.50", annotation_position="top left")

        fig_pr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.1)",
            font=dict(color="#cbd5e1"),
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(title="Decision Threshold", gridcolor="#23293e"),
            yaxis=dict(title="Score (0.0 to 1.0)", gridcolor="#23293e", range=[0, 1.05]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # 4. Feature Impact: Log-Odds & Random Forest Gini Importance
    st.markdown("### 4. Feature Importance & Interpretability")
    fi_col1, fi_col2 = st.columns(2)

    with fi_col1:
        st.markdown("##### Logistic Regression Odds Ratios (Balanced)")
        st.caption("Odds ratio > 1.0 indicates higher smoking likelihood; < 1.0 indicates protective/lower likelihood.")

        impact_df = suite["feature_impact_df"].copy()
        top_impact = pd.concat([impact_df.head(8), impact_df.tail(6)]).drop_duplicates()
        top_impact["direction"] = top_impact["lr_balanced_odds_ratio"].apply(
            lambda x: "Increases Odds" if x > 1.0 else "Decreases Odds"
        )

        fig_or = px.bar(
            top_impact.sort_values(by="lr_balanced_odds_ratio", ascending=True),
            x="lr_balanced_odds_ratio",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={"Increases Odds": "#ff2a5f", "Decreases Odds": "#10b981"},
            labels={"lr_balanced_odds_ratio": "Odds Ratio (exp(β))", "feature": "Feature"},
            text="lr_balanced_odds_ratio"
        )
        fig_or.add_vline(x=1.0, line_dash="dash", line_color="#cbd5e1")
        fig_or.update_traces(texttemplate='%{text:.2f}x', textposition='outside')
        fig_or.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.1)",
            font=dict(color="#cbd5e1"),
            height=380,
            margin=dict(l=10, r=20, t=10, b=10),
            xaxis=dict(gridcolor="#23293e"),
            yaxis=dict(gridcolor="#23293e"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_or, use_container_width=True)

    with fi_col2:
        st.markdown("##### Random Forest Gini Importance")
        st.caption("Mean decrease in impurity across 100 decision trees (max_depth=5).")

        top_rf = impact_df.sort_values(by="rf_importance", ascending=True).tail(12)
        fig_rf = px.bar(
            top_rf,
            x="rf_importance",
            y="feature",
            orientation="h",
            color="rf_importance",
            color_continuous_scale=[[0, "#4c1d95"], [1, "#06b6d4"]],
            labels={"rf_importance": "Gini Importance", "feature": "Feature"}
        )
        fig_rf.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.1)",
            font=dict(color="#cbd5e1"),
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#23293e"),
            yaxis=dict(gridcolor="#23293e"),
        )
        st.plotly_chart(fig_rf, use_container_width=True)

    # 5. Production Engineering Architecture & Code Deep Dives
    st.markdown("### 5. Pipeline Architecture & Code Breakdown")

    with st.expander("🛠️ Data Engineering: Fixing the 'Did Not Answer' Flaw", expanded=False):
        drug_rates, drug_counts = answer_rates(df_raw, "drugs")
        st.markdown(f"""
In the exploratory notebook, categorical missing values were left as `NaN` and dummy-encoded with `drop_first=True`.
This caused missing entries to encode as all-zeros, **silently folding 'did not answer' into 'never' (the dropped reference category)**.
Two rows that mean opposite things became one row of numbers.

**The fix in `smoking_model.py`:**
```python
# An explicit category preserves the signal that a question was avoided
drugs_series = df_clean["drugs"].apply(lambda x: x if x in DRUG_OPTIONS else "unspecified")
```

On the dataset this instance loaded, profiles that withheld their drug usage smoke at
**{drug_rates.get('unspecified', float('nan')):.1%}** ({drug_counts.get('unspecified', 0):,} profiles),
against **{drug_rates.get('never', float('nan')):.1%}** for those who answered `never`.
The notebook's encoding made that distinction unavailable to the model.
""")

    with st.expander("📐 Mathematical Formulation: Balanced Cross-Entropy Loss", expanded=False):
        # Plain string, not an f-string: the LaTeX braces below would otherwise be read
        # as replacement fields.
        st.markdown("""
        For class-balanced Logistic Regression, scikit-learn weights each training sample $i$ belonging to class $j \\in \\{0, 1\\}$:
        $$\\text{Loss} = - \\sum_{i=1}^N w_{y_i} \\left[ y_i \\log(p_i) + (1 - y_i) \\log(1 - p_i) \\right]$$
        where the class weight is computed as:
        $$w_j = \\frac{N}{2 \\cdot N_j}$$
        """)
        # The weights are derived from the class balance of the split that was actually
        # used, so this sentence survives a change of dataset.
        st.markdown(
            f"With the {train_rows:,} training rows this model was fitted on and a smoker "
            f"share of {base_rate:.1%}, balanced weighting comes out at "
            f"$w_0 \\approx {1 / (2 * (1 - base_rate)):.2f}$ for non-smokers and "
            f"$w_1 \\approx {1 / (2 * base_rate):.2f}$ for smokers, so the optimizer treats a "
            f"missed smoker as roughly "
            f"**{(1 / (2 * base_rate)) / (1 / (2 * (1 - base_rate))):.2f}x more costly** than a "
            "false positive on a non-smoker."
        )

    with st.expander("💻 Modular Pipeline Code (`smoking_model.py`)", expanded=False):
        st.markdown("""
        The complete pipeline is decoupled from Streamlit and testable standalone:
        - `prepare_training_data()`: Generates one-hot feature matrix with schema guarantees.
        - `train_model_suite()`: Fits 5 benchmark models, calculates metrics, confusion matrices, and ROC-AUC.
        - `predict_single_profile()`: Exposes a single-sample inference endpoint with log-odds attributions and risk categorization.
        - `compute_threshold_curve()`: Pre-computes precision/recall curves for production calibration.
        """)

# ---------------------------------------------------------
# TAB 4: BAY AREA 2012 DATA STUDIO (THE WHOLE SHEBANG)
# ---------------------------------------------------------
elif mode == "Bay Area 2012 Data Studio":
    st.markdown('<div class="hero-title">Bay Area 2012 Data Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Interactive exploration of demographics, habits, careers, and geography in the 2012 Silicon Valley & Bay Area dating ecosystem.</div>', unsafe_allow_html=True)

    tab_demo, tab_vices, tab_work, tab_faith, tab_geo = st.tabs([
        "Demographics & Dating Pool",
        "Habits & Lifestyle",
        "Tech Boom & Careers",
        "Faith & Astrological Signs",
        "Bay Area Geography"
    ])

    # 1. Demographics Tab
    with tab_demo:
        st.markdown("### Age, Orientation & Dating Pool Dynamics")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            # Age distribution by sex
            age_clean = df_raw.dropna(subset=['age', 'sex'])
            fig_age = px.histogram(
                age_clean,
                x="age",
                color="sex",
                barmode="overlay",
                nbins=40,
                color_discrete_map={"m": "#38bdf8", "f": "#ff2a5f"},
                title="Age Distribution of OkCupid Daters in 2012 (Men vs Women)",
                labels={"age": "Age", "sex": "Gender"}
            )
            fig_age.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(fig_age, use_container_width=True)

        with col_d2:
            # Orientation breakdown
            orient_counts = df_raw['orientation'].value_counts().reset_index()
            orient_counts.columns = ['Orientation', 'Count']
            fig_orient = px.pie(
                orient_counts,
                values="Count",
                names="Orientation",
                hole=0.55,
                color_discrete_sequence=["#ff2a5f", "#9055ff", "#06b6d4"],
                title="Sexual Orientation Breakdown (SF Bay Area 2012)"
            )
            fig_orient.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(fig_orient, use_container_width=True)

        col_d3, col_d4 = st.columns(2)
        with col_d3:
            # Height distribution
            height_clean = df_raw.dropna(subset=['height', 'sex'])
            height_clean = height_clean[(height_clean['height'] >= 55) & (height_clean['height'] <= 80)]
            fig_height = px.box(
                height_clean,
                x="sex",
                y="height",
                color="sex",
                color_discrete_map={"m": "#38bdf8", "f": "#ff2a5f"},
                title="Reported Height Distribution by Sex (Inches)",
                labels={"height": "Height (in)", "sex": "Gender"}
            )
            fig_height.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(fig_height, use_container_width=True)

        with col_d4:
            # Relationship status
            status_counts = df_raw['status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            fig_status = px.bar(
                status_counts,
                x="Status",
                y="Count",
                color="Status",
                color_discrete_sequence=px.colors.sequential.Sunsetdark,
                title="Relationship Status of Profiles"
            )
            fig_status.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(fig_status, use_container_width=True)

    # 2. Vices / Habits Tab
    with tab_vices:
        st.markdown("### Drinking, Smoking & Drug Habits")

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            # Clear bar chart of drinking habits
            drink_counts = df_raw[df_raw['drinks'] != 'unspecified']['drinks'].value_counts().reset_index()
            drink_counts.columns = ['Habit', 'Profiles']
            fig_drinks = px.bar(
                drink_counts,
                x="Habit",
                y="Profiles",
                color="Habit",
                color_discrete_sequence=px.colors.sequential.Teal,
                title="Drinking Habits Distribution"
            )
            fig_drinks.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(fig_drinks, use_container_width=True)

        with col_v2:
            # Clear bar chart of smoking habits
            smoke_counts = df_raw[df_raw['smokes'] != 'unspecified']['smokes'].value_counts().reset_index()
            smoke_counts.columns = ['Habit', 'Profiles']
            fig_smokes = px.bar(
                smoke_counts,
                x="Habit",
                y="Profiles",
                color="Habit",
                color_discrete_sequence=px.colors.sequential.Burg,
                title="Smoking Habits Distribution"
            )
            fig_smokes.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(fig_smokes, use_container_width=True)

        st.markdown("##### Drug Usage Patterns Across Age Cohorts")
        df_temp = df_raw.dropna(subset=['age', 'drugs']).copy()
        df_temp = df_temp[df_temp['drugs'] != 'unspecified']
        df_temp['age_group'] = pd.cut(df_temp['age'], bins=[18, 25, 30, 40, 50, 70], labels=['18-24', '25-29', '30-39', '40-49', '50+'])
        drug_age = df_temp.groupby(['age_group', 'drugs'], observed=False).size().reset_index(name='count')
        
        fig_drug = px.bar(
            drug_age,
            x="age_group",
            y="count",
            color="drugs",
            barmode="stack",
            color_discrete_map={"never": "#10b981", "sometimes": "#f59e0b", "often": "#ef4444"},
            labels={"age_group": "Age Group", "count": "Profiles", "drugs": "Drugs"}
        )
        fig_drug.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1")
        )
        st.plotly_chart(fig_drug, use_container_width=True)

    # 3. Work & Income Tab
    with tab_work:
        st.markdown("### Silicon Valley 2012: Careers & Education")

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            job_counts = df_raw[df_raw['job'] != 'unspecified']['job'].value_counts().head(12).reset_index()
            job_counts.columns = ['Profession', 'Count']
            fig_jobs = px.bar(
                job_counts,
                y="Profession",
                x="Count",
                orientation="h",
                color="Count",
                color_continuous_scale="Viridis",
                title="Top 12 Occupations in 2012 SF Bay Dating Pool"
            )
            fig_jobs.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(fig_jobs, use_container_width=True)

        with col_w2:
            edu_counts = df_raw['edu_category'].value_counts().reset_index()
            edu_counts.columns = ['Education Level', 'Count']
            fig_edu = px.pie(
                edu_counts,
                names="Education Level",
                values="Count",
                color_discrete_sequence=px.colors.sequential.Plasma,
                title="Education Level Breakdown of Bay Area Daters",
                hole=0.45
            )
            fig_edu.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(fig_edu, use_container_width=True)

        # Reported Income
        income_df = df_raw[(df_raw['income'] > 0) & (df_raw['income'] < 1000000)].copy()
        if not income_df.empty:
            st.markdown("##### Self-Reported Incomes (Profiles who disclosed)")
            fig_income = px.histogram(
                income_df,
                x="income",
                nbins=30,
                color_discrete_sequence=["#10b981"],
                title="Self-Reported Annual Income ($USD in 2012)",
                labels={"income": "Annual Income ($)"}
            )
            fig_income.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(fig_income, use_container_width=True)

    # 4. Faith & Zodiac Tab
    with tab_faith:
        st.markdown("### Faith, Seriousness & Astrology in California")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            rel_counts = df_raw[df_raw['religion_base'] != 'unspecified']['religion_base'].value_counts().reset_index()
            rel_counts.columns = ['Faith', 'Count']
            fig_rel = px.bar(
                rel_counts,
                x="Faith",
                y="Count",
                color="Faith",
                color_discrete_sequence=px.colors.qualitative.Prism,
                title="Primary Worldview / Religion"
            )
            fig_rel.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(fig_rel, use_container_width=True)

        with col_f2:
            ser_counts = df_raw[df_raw['religion_seriousness'] != 'unspecified']['religion_seriousness'].value_counts().reset_index()
            ser_counts.columns = ['Attitude', 'Count']
            fig_ser = px.pie(
                ser_counts,
                names="Attitude",
                values="Count",
                hole=0.5,
                color_discrete_sequence=["#a855f7", "#ec4899", "#3b82f6", "#14b8a6", "#f59e0b"],
                title="How Seriously People Take Their Beliefs"
            )
            fig_ser.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(fig_ser, use_container_width=True)

        # Zodiac Sign Breakdown
        astro_df = df_raw[df_raw['astrology_sign'] != 'unspecified']['astrology_sign'].value_counts().reset_index()
        astro_df.columns = ['Zodiac Sign', 'Count']
        fig_astro = px.bar(
            astro_df,
            x="Zodiac Sign",
            y="Count",
            color="Count",
            color_continuous_scale="Reds",
            title="Distribution of Astrological Signs"
        )
        fig_astro.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1")
        )
        st.plotly_chart(fig_astro, use_container_width=True)

    # 5. Geography Tab
    with tab_geo:
        st.markdown("### Geography: Bay Area Cities")
        
        city_counts = df_raw[df_raw['city'] != 'other']['city'].value_counts().head(15).reset_index()
        city_counts.columns = ['City', 'Profiles']
        
        fig_cities = px.bar(
            city_counts,
            x="Profiles",
            y="City",
            orientation="h",
            color="Profiles",
            color_continuous_scale="Magenta",
            title="Top 15 Northern California Cities in Dataset"
        )
        fig_cities.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1")
        )
        st.plotly_chart(fig_cities, use_container_width=True)

# ---------------------------------------------------------
# TAB 5: CUPID'S PROFILE DETECTIVE
# ---------------------------------------------------------
elif mode == "Profile Detective":
    st.markdown('<div class="hero-title">Profile Detective</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Search and read authentic 2012 dating essays and self-descriptions.</div>', unsafe_allow_html=True)

    filter_c1, filter_c2, filter_c3 = st.columns([2, 1, 1])
    with filter_c1:
        keyword = st.text_input("Search bios by keyword:", placeholder="e.g. startup, hiking, sushi, radiohead, coffee, books")
    with filter_c2:
        filter_gender = st.selectbox("Gender:", ["All", "Women", "Men"])
    with filter_c3:
        filter_city = st.selectbox("City:", ["All Cities", "San Francisco", "Oakland", "Berkeley", "San Jose"])

    # Filter detective profiles
    det_df = df_raw.copy()
    if filter_gender == "Women":
        det_df = det_df[det_df['sex'] == 'f']
    elif filter_gender == "Men":
        det_df = det_df[det_df['sex'] == 'm']

    if filter_city != "All Cities":
        det_df = det_df[det_df['city'] == filter_city.lower()]

    if keyword.strip():
        k = keyword.lower().strip()
        mask = (
            det_df['essay0'].str.lower().str.contains(k, na=False) |
            det_df['essay1'].str.lower().str.contains(k, na=False) |
            det_df['essay2'].str.lower().str.contains(k, na=False) |
            det_df['essay4'].str.lower().str.contains(k, na=False) |
            det_df['essay9'].str.lower().str.contains(k, na=False)
        )
        det_df = det_df[mask]

    st.markdown(f"Found **{len(det_df):,}** profiles matching search criteria.")

    if len(det_df) > 0:
        if "det_seed" not in st.session_state:
            st.session_state.det_seed = 101

        if st.button("Pull Another Random Profile"):
            st.session_state.det_seed = np.random.randint(1, 100000)
            st.rerun()

        chosen_profile = det_df.sample(n=1, random_state=st.session_state.det_seed).iloc[0]

        # Profile Overview Card
        age = int(chosen_profile['age']) if not pd.isna(chosen_profile['age']) else "?"
        sex = "Female" if chosen_profile['sex'] == 'f' else "Male"
        height = f"{int(chosen_profile['height'] // 12)}'{int(chosen_profile['height'] % 12)}\"" if not pd.isna(chosen_profile['height']) else "?"
        job = chosen_profile['job'].title() if chosen_profile['job'] != 'unspecified' else "Not specified"
        city = chosen_profile['city'].title()
        sign = chosen_profile['astrology_sign'].capitalize()

        st.markdown(f"""
        <div class="profile-card" style="border: 2px solid #ff2a5f;">
            <div class="profile-header" style="font-size: 1.5rem; color: #ff2a5f;">{age} yr old • {sex} in {city}, CA</div>
            <div style="margin-top: 0.5rem;">
                <span class="badge badge-pink">Height: {height}</span>
                <span class="badge badge-purple">Career: {job}</span>
                <span class="badge badge-cyan">Orientation: {chosen_profile['orientation'].capitalize()}</span>
                <span class="badge badge-amber">Sign: {sign}</span>
                <span class="badge badge-emerald">Body: {chosen_profile['body_type'].capitalize()}</span>
            </div>
            <div style="margin-top: 0.8rem; font-size: 0.95rem; color: #cbd5e1;">
                Drinks: <b>{chosen_profile['drinks'].capitalize()}</b> | 
                Smokes: <b>{chosen_profile['smokes'].capitalize()}</b> | 
                Drugs: <b>{chosen_profile['drugs'].capitalize()}</b> | 
                Religion: <b>{chosen_profile['religion_base'].capitalize()}</b> ({chosen_profile['religion_seriousness']})
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Essays Display
        essay_prompts = [
            ("essay0", "My self-summary"),
            ("essay1", "What I’m doing with my life"),
            ("essay2", "I’m really good at"),
            ("essay3", "The first thing people usually notice about me"),
            ("essay4", "Favorite books, movies, shows, music, and food"),
            ("essay5", "Six things I could never do without"),
            ("essay6", "I spend a lot of time thinking about"),
            ("essay7", "On a typical Friday night I am"),
            ("essay8", "The most private thing I’m willing to admit"),
            ("essay9", "You should message me if")
        ]

        st.markdown("#### OkCupid Dating Essays (2012 Original Text)")
        for key, title in essay_prompts:
            raw_text = chosen_profile.get(key, '')
            clean_text = clean_html(raw_text)
            if clean_text and len(clean_text) > 3:
                with st.expander(title, expanded=(key in ["essay0", "essay4", "essay9"])):
                    st.write(clean_text)
    else:
        st.warning("No profiles match this keyword or filter. Try searching for words like 'music', 'food', 'hiking', or clear the keyword.")


