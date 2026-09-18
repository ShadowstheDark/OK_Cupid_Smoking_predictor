from typing import Dict, Any, Optional, List
import os
import re
import html
import pandas as pd
import numpy as np

# Path resolution
#
# Three sources are tried, in order of preference:
#   1. $OKCUPID_CSV              - explicit override, any path on disk
#   2. data/full/profiles.csv    - the complete 59,946-row extract (git-ignored)
#   3. data/sample_profiles.csv  - the committed 10,000-row demo sample
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_DATASET_PATH = os.path.join(BASE_DIR, "data", "full", "profiles.csv")
SAMPLE_DATASET_PATH = os.path.join(BASE_DIR, "data", "sample_profiles.csv")

# Row count of the complete 2012 extract, used to label the demo sample and to
# show how much of the full population is being analysed.
FULL_DATASET_ROWS = 59946

# Canonical education buckets produced by clean_education_category(). The UI
# builds its filter options from this list so the two can never drift apart.
EDUCATION_CATEGORIES: List[str] = [
    "College Graduate",
    "Master's Degree",
    "Post-Graduate / Ph.D",
    "In College / Associate",
    "High School",
    "Space Camp",
    "College Dropout",
    "Other / In Progress",
]


def get_csv_path() -> str:
    """Return the dataset to load, preferring the full extract over the sample."""
    override = os.environ.get("OKCUPID_CSV")
    candidates = [override] if override else []
    candidates += [FULL_DATASET_PATH, SAMPLE_DATASET_PATH]

    for path in candidates:
        if path and os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No OkCupid dataset found. Looked for:\n"
        + "\n".join("  - " + str(p) for p in candidates)
        + "\n\nKeep data/sample_profiles.csv next to this file, or download the full "
        "profiles.csv and save it to data/full/profiles.csv. See DATA.md."
    )


def using_sample() -> bool:
    """True when we are running on the committed demo sample, not the full extract."""
    return os.path.abspath(get_csv_path()) == os.path.abspath(SAMPLE_DATASET_PATH)

def clean_html(raw_text: Any) -> str:
    if not isinstance(raw_text, str) or pd.isna(raw_text):
        return ""
    # Unescape HTML entities (&amp;, &rsquo;, etc.)
    text = html.unescape(raw_text)
    # Replace <br />, <br>, <br/> with newline
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    # Remove any other HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Strip excess whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_religion_base(val: Any) -> str:
    if pd.isna(val) or not isinstance(val, str):
        return "unspecified"
    val = val.lower().strip()
    for rel in ["agnosticism", "atheism", "christianity", "catholicism", "judaism", "buddhism", "hinduism", "islam"]:
        if rel in val:
            return rel
    if "other" in val:
        return "other"
    return "unspecified"

def extract_religion_seriousness(val: Any) -> str:
    if pd.isna(val) or not isinstance(val, str):
        return "unspecified"
    val = val.lower().strip()
    if "very serious" in val:
        return "very serious"
    if "somewhat serious" in val:
        return "somewhat serious"
    if "not too serious" in val:
        return "not too serious"
    if "laughing about it" in val:
        return "laughing about it"
    if val != "unspecified" and val != "":
        return "matter-of-fact"
    return "unspecified"

def extract_city(val: Any) -> str:
    if pd.isna(val) or not isinstance(val, str):
        return "other"
    parts = [p.strip() for p in val.lower().split(",")]
    if len(parts) >= 1:
        return parts[0]
    return "other"

def extract_pets(val: Any) -> Dict[str, Optional[bool]]:
    if pd.isna(val) or not isinstance(val, str):
        return {"likes_dogs": None, "likes_cats": None, "has_dogs": None, "has_cats": None}
    val = val.lower()
    return {
        "likes_dogs": "likes dogs" in val or "has dogs" in val,
        "likes_cats": "likes cats" in val or "has cats" in val,
        "has_dogs": "has dogs" in val,
        "has_cats": "has cats" in val
    }

def clean_education_category(val: Any) -> str:
    if pd.isna(val) or not isinstance(val, str):
        return "unspecified"
    val = val.lower().strip()
    if "ph.d" in val or "postgrad" in val:
        return "Post-Graduate / Ph.D"
    if "masters" in val or "master" in val:
        return "Master's Degree"
    if "graduated from college" in val or ("college/university" in val and "graduated" in val):
        return "College Graduate"
    if "working on college" in val or "two-year college" in val:
        return "In College / Associate"
    if "high school" in val:
        return "High School"
    if "space camp" in val:
        # Deliberately no emoji here: the label is used as a filter key by the
        # UI and shows up in logs and terminals, and a non-ASCII emoji broke
        # both. "Space Camp" is what the app offers in its dropdown.
        return "Space Camp"
    if "dropped out" in val:
        return "College Dropout"
    return "Other / In Progress"

def load_data() -> pd.DataFrame:
    csv_path = get_csv_path()
    df = pd.read_csv(csv_path)

    # Standardize age and height
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['height'] = pd.to_numeric(df['height'], errors='coerce')
    
    # Filter out extreme anomalies
    df.loc[(df['age'] < 18) | (df['age'] > 85), 'age'] = np.nan
    df.loc[(df['height'] < 50) | (df['height'] > 86), 'height'] = np.nan

    # Clean text columns
    str_cols = ['sex', 'orientation', 'drinks', 'smokes', 'drugs', 'body_type', 'education', 'job', 'status', 'pets', 'location']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unspecified').astype(str).str.strip().str.lower()

    # Extracted fields
    df['religion_base'] = df['religion'].apply(extract_religion_base)
    df['religion_seriousness'] = df['religion'].apply(extract_religion_seriousness)
    df['city'] = df['location'].apply(extract_city)
    df['edu_category'] = df['education'].apply(clean_education_category)

    # Pet preferences
    pet_info = df['pets'].apply(extract_pets)
    pet_df = pd.DataFrame(pet_info.tolist())
    df['likes_dogs'] = pet_df['likes_dogs']
    df['likes_cats'] = pet_df['likes_cats']
    df['has_dogs'] = pet_df['has_dogs']
    df['has_cats'] = pet_df['has_cats']

    # Height in cm
    df['height_cm'] = (df['height'] * 2.54).round(1)

    # Astrological sign base
    df['astrology_sign'] = df['sign'].apply(lambda s: s.split()[0] if isinstance(s, str) and not pd.isna(s) else 'unspecified')

    return df

if __name__ == "__main__":
    print("Testing data loader...")
    df = load_data()
    print(f"Loaded {len(df):,} rows successfully!")
    print("Religion bases:", df['religion_base'].value_counts().to_dict())
    print("Top 5 Cities:", df['city'].value_counts().head(5).to_dict())
    print("Data loader test passed!")
