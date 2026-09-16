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

# ---------------------------------------------------------
# Page Configuration & Custom Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cupid 2012 | California Dating Odds & Data Studio",
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

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
# Each mode is addressable, so a link can open straight into one of them:
#   ?page=calculator   ?page=studio   ?page=detective
PAGE_OPTIONS = [
    "Match Calculator (Reality Check)",
    "Bay Area 2012 Data Studio",
    "Profile Detective",
]
PAGE_SLUGS = {"calculator": 0, "studio": 1, "detective": 2}

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

    # ---------------------------------------------------------
    # Calculation & Funnel Simulation
    # ---------------------------------------------------------
    current_df = df_raw.copy()
    funnel_steps = [("All 2012 Profiles", len(current_df))]

    # 1. Gender Filter
    if pref_gender == "Women":
        current_df = current_df[current_df['sex'] == 'f']
    elif pref_gender == "Men":
        current_df = current_df[current_df['sex'] == 'm']
    funnel_steps.append(("Gender Filter", len(current_df)))

    # 2. Orientation
    if pref_orientation:
        current_df = current_df[current_df['orientation'].isin(pref_orientation)]
    funnel_steps.append(("Orientation", len(current_df)))

    # 3. Age Range
    current_df = current_df[(current_df['age'] >= age_min) & (current_df['age'] <= age_max)]
    funnel_steps.append(("Age Window", len(current_df)))

    # 4. Status
    if pref_status:
        current_df = current_df[current_df['status'].isin(pref_status)]
    funnel_steps.append(("Status", len(current_df)))

    # 5. Height
    current_df = current_df[(current_df['height'] >= height_min) & (current_df['height'] <= height_max)]
    funnel_steps.append(("Height Range", len(current_df)))

    # 6. Smoking
    if pref_smokes:
        current_df = current_df[current_df['smokes'].isin(pref_smokes) | (current_df['smokes'] == 'unspecified')]
    funnel_steps.append(("Smoking Habits", len(current_df)))

    # 7. Drinking
    if pref_drinks:
        current_df = current_df[current_df['drinks'].isin(pref_drinks) | (current_df['drinks'] == 'unspecified')]
    funnel_steps.append(("Drinking Habits", len(current_df)))

    # 8. Drugs
    if pref_drugs:
        current_df = current_df[current_df['drugs'].isin(pref_drugs) | (current_df['drugs'] == 'unspecified')]
    funnel_steps.append(("Drug Habits", len(current_df)))

    # 9. Religion
    if pref_rel:
        current_df = current_df[current_df['religion_base'].isin(pref_rel)]
    if pref_seriousness:
        current_df = current_df[current_df['religion_seriousness'].isin(pref_seriousness)]
    funnel_steps.append(("Religion / Beliefs", len(current_df)))

    # 10. Education
    if pref_edu:
        current_df = current_df[current_df['edu_category'].isin(pref_edu)]
    funnel_steps.append(("Education", len(current_df)))

    # 11. Pets
    if pref_pets == "Must Like/Have Dogs":
        current_df = current_df[current_df['likes_dogs'] == True]
    elif pref_pets == "Must Like/Have Cats":
        current_df = current_df[current_df['likes_cats'] == True]
    elif pref_pets == "Must Like Both Dogs & Cats":
        current_df = current_df[(current_df['likes_dogs'] == True) & (current_df['likes_cats'] == True)]
    funnel_steps.append(("Pet Preference", len(current_df)))

    # 12. Location
    if pref_city != "All Bay Area":
        current_df = current_df[current_df['city'] == pref_city]
    funnel_steps.append(("Location", len(current_df)))

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
        - **Dealbreaker Funnel**: Sequentially applies each criterion and measures how many candidates remain at each stage, as well as the marginal percentage dropped.
        """)

    # ---------------------------------------------------------
    # Visual Dealbreaker Funnel
    # ---------------------------------------------------------
    f_col1, f_col2 = st.columns([3, 2])

    with f_col1:
        st.markdown("#### The Dealbreaker Funnel")
        st.caption("Attrition across consecutive filters:")

        funnel_df = pd.DataFrame(funnel_steps, columns=["Step", "Remaining"])
        
        # Calculate percentage drop at each step
        funnel_df["Drop"] = funnel_df["Remaining"].shift(1) - funnel_df["Remaining"]
        funnel_df["Drop_Pct"] = (funnel_df["Drop"] / funnel_df["Remaining"].shift(1)) * 100
        funnel_df["Drop_Pct"] = funnel_df["Drop_Pct"].fillna(0)

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

    with f_col2:
        st.markdown("#### Dealbreaker Breakdown")
        st.caption("Largest single factor reducing candidates:")

        drops = funnel_df[funnel_df["Drop"] > 0].sort_values(by="Drop_Pct", ascending=False)
        if not drops.empty:
            biggest = drops.iloc[0]
            st.warning(f"**Biggest Bottleneck: {biggest['Step']}**\n\nEliminated **{biggest['Drop_Pct']:.1f}%** ({int(biggest['Drop']):,} candidates) of the remaining pool.")
            
            st.markdown("##### Other Major Reductions:")
            for idx, row in drops.iloc[1:5].iterrows():
                st.markdown(f"- **{row['Step']}**: eliminated **{row['Drop_Pct']:.1f}%** ({int(row['Drop']):,} candidates)")
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
# TAB 2: BAY AREA 2012 DATA STUDIO (THE WHOLE SHEBANG)
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
# TAB 3: CUPID'S PROFILE DETECTIVE
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


