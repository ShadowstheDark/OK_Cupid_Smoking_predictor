"""
tests/test_data_loader.py
=========================
Unit tests for data sanitization and feature extraction in data_loader.py.
"""

import os
import pytest
import pandas as pd
import numpy as np
from data_loader import (
    clean_html,
    extract_pets,
    extract_religion_base,
    extract_religion_seriousness,
    extract_city,
    clean_education_category,
    load_data,
    EDUCATION_CATEGORIES,
)


class TestDataSanitization:
    """Test string cleaning and regex-based HTML stripping."""

    def test_clean_html_strips_tags(self):
        raw = "Hello <b>world</b>! How are you?<br />I'm good."
        cleaned = clean_html(raw)
        assert "<b>" not in cleaned
        assert "</b>" not in cleaned
        assert "<br />" not in cleaned
        assert "Hello world! How are you?\nI'm good." == cleaned

    def test_clean_html_unescapes_entities(self):
        raw = "Cats &amp; dogs &rsquo; love &quot;fun&quot;"
        cleaned = clean_html(raw)
        assert "&amp;" not in cleaned
        assert "Cats & dogs ’ love \"fun\"" == cleaned

    def test_clean_html_handles_nan_and_non_strings(self):
        assert clean_html(None) == ""
        assert clean_html(np.nan) == ""
        assert clean_html(123) == ""


class TestFeatureExtractors:
    """Test domain-specific extraction functions."""

    def test_extract_pets_positive(self):
        info = extract_pets("likes dogs and has cats")
        assert info["likes_dogs"] is True
        assert info["likes_cats"] is True
        assert info["has_dogs"] is False
        assert info["has_cats"] is True

    def test_extract_pets_none_for_missing(self):
        info = extract_pets(np.nan)
        assert info["likes_dogs"] is None
        assert info["likes_cats"] is None

    def test_extract_religion_base(self):
        assert extract_religion_base("agnosticism and laughing about it") == "agnosticism"
        assert extract_religion_base("catholicism but not too serious") == "catholicism"
        assert extract_religion_base("buddhism") == "buddhism"
        assert extract_religion_base(np.nan) == "unspecified"
        assert extract_religion_base("pastafarian") == "unspecified"

    def test_extract_religion_seriousness(self):
        assert extract_religion_seriousness("christianity and somewhat serious") == "somewhat serious"
        assert extract_religion_seriousness("judaism and laughing about it") == "laughing about it"
        assert extract_religion_seriousness("atheism") == "matter-of-fact"
        assert extract_religion_seriousness(np.nan) == "unspecified"

    def test_extract_city(self):
        assert extract_city("san francisco, california") == "san francisco"
        assert extract_city("oakland, california") == "oakland"
        assert extract_city(np.nan) == "other"

    def test_clean_education_category_precedence_and_mapping(self):
        # Testing the operator precedence fix on line 124
        assert clean_education_category("graduated from college/university") == "College Graduate"
        assert clean_education_category("graduated from college") == "College Graduate"
        assert clean_education_category("working on college/university") == "In College / Associate"
        assert clean_education_category("masters program") == "Master's Degree"
        assert clean_education_category("ph.d program") == "Post-Graduate / Ph.D"
        assert clean_education_category("space camp") == "Space Camp"
        assert clean_education_category("dropped out of high school") == "High School"
        assert clean_education_category("dropped out of college/university") == "College Dropout"
        assert clean_education_category(np.nan) == "unspecified"

    def test_education_categories_consistency(self):
        # Ensure cleaned buckets belong to canonical list or unspecified
        samples = [
            "graduated from college/university",
            "ph.d",
            "working on two-year college",
            "high school",
            "space camp",
            "masters",
        ]
        for s in samples:
            cat = clean_education_category(s)
            assert cat in EDUCATION_CATEGORIES or cat == "unspecified"


class TestDatasetIntegrity:
    """Test full dataset loader behavior."""

    @pytest.fixture(scope="class")
    def dataset(self):
        return load_data()

    def test_dataset_not_empty(self, dataset):
        assert len(dataset) > 0

    def test_required_columns_exist(self, dataset):
        required_cols = [
            "age", "height", "sex", "orientation", "drinks", "smokes",
            "drugs", "body_type", "education", "religion_base", "city",
            "likes_dogs", "likes_cats"
        ]
        for col in required_cols:
            assert col in dataset.columns, f"Missing required column: {col}"

    def test_age_and_height_ranges(self, dataset):
        valid_ages = dataset["age"].dropna()
        assert (valid_ages >= 18).all()
        assert (valid_ages <= 85).all()

        valid_heights = dataset["height"].dropna()
        assert (valid_heights >= 50).all()
        assert (valid_heights <= 86).all()
