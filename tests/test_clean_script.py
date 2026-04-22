import pytest
import pandas as pd
import datetime

# ── helpers ──────────────────────────────────────────────────────────────────

REPLACEMENTS = {
    "Armed conflicts and attacks": "armed conflicts and attacks",
    "Disasters and accidents": "disasters and accidents",
    "Law and crime": "law and crime and politics",
    "Politics and elections": "politics and elections and economics",
    "International relations": "international relations",
    "Business and economy": "business and economics",
    "Sports": "sports",
    "Science and technology": "science and technology",
    "Health and environment": "health and environment",
    "Arts and culture": "arts and culture",
    # Typos / variants
    "Science and Technology": "science and technology",
    "Disaster and accidents": "disasters and accidents",
    "Arts and Culture": "arts and culture",
    "Business and econony": "business and economics",
    "Attacks and armed conflicts": "armed conflicts and attacks",
}

VALID_TOPICS = {
    "armed conflicts and attacks",
    "law and crime and politics",
    "disasters and accidents",
    "politics and elections and economics",
    "international relations",
    "health and environment",
    "business and economics",
    "sports",
    "science and technology",
    "arts and culture",
}


def apply_topic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the topic-cleaning logic in the main script."""
    df = df.copy()
    df["topic"] = df["topic"].replace(REPLACEMENTS, regex=False)
    df["topic"] = df["topic"].str.lower()
    return df


def apply_text_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the text-cleaning logic in the main script."""
    df = df.copy()
    df["text"] = df["text"].str.replace("^edit,history,watch", "", regex=True)
    df["text"] = df["text"].str.replace("\n", "", regex=False)
    return df


# ── topic replacement tests ───────────────────────────────────────────────────

class TestTopicReplacements:

    def test_standard_capitalised_topics_are_lowercased(self):
        """Original capitalised topics map to their lowercase equivalents."""
        df = pd.DataFrame({"topic": list(REPLACEMENTS.keys())})
        result = apply_topic_cleaning(df)
        assert all(t in VALID_TOPICS for t in result["topic"])

    def test_law_and_crime_maps_to_law_and_crime_and_politics(self):
        df = pd.DataFrame({"topic": ["Law and crime"]})
        result = apply_topic_cleaning(df)
        assert result["topic"].iloc[0] == "law and crime and politics"

    def test_politics_and_elections_maps_to_with_economics(self):
        df = pd.DataFrame({"topic": ["Politics and elections"]})
        result = apply_topic_cleaning(df)
        assert result["topic"].iloc[0] == "politics and elections and economics"

    def test_business_and_economy_maps_to_business_and_economics(self):
        df = pd.DataFrame({"topic": ["Business and economy"]})
        result = apply_topic_cleaning(df)
        assert result["topic"].iloc[0] == "business and economics"

    def test_typo_business_econony_is_corrected(self):
        df = pd.DataFrame({"topic": ["Business and econony"]})
        result = apply_topic_cleaning(df)
        assert result["topic"].iloc[0] == "business and economics"

    def test_typo_disaster_singular_is_corrected(self):
        df = pd.DataFrame({"topic": ["Disaster and accidents"]})
        result = apply_topic_cleaning(df)
        assert result["topic"].iloc[0] == "disasters and accidents"

    def test_variant_attacks_and_armed_conflicts_is_normalised(self):
        df = pd.DataFrame({"topic": ["Attacks and armed conflicts"]})
        result = apply_topic_cleaning(df)
        assert result["topic"].iloc[0] == "armed conflicts and attacks"

    def test_science_and_technology_mixed_case_is_normalised(self):
        df = pd.DataFrame({"topic": ["Science and Technology"]})
        result = apply_topic_cleaning(df)
        assert result["topic"].iloc[0] == "science and technology"

    def test_arts_and_culture_mixed_case_is_normalised(self):
        df = pd.DataFrame({"topic": ["Arts and Culture"]})
        result = apply_topic_cleaning(df)
        assert result["topic"].iloc[0] == "arts and culture"

    def test_all_output_topics_are_in_valid_set(self):
        """After cleaning, every topic must be one of the 10 canonical values."""
        df = pd.DataFrame({"topic": list(REPLACEMENTS.keys())})
        result = apply_topic_cleaning(df)
        invalid = set(result["topic"]) - VALID_TOPICS
        assert not invalid, f"Unexpected topics after cleaning: {invalid}"

    def test_already_clean_topics_are_unchanged(self):
        """Topics already in the correct form should pass through untouched."""
        df = pd.DataFrame({"topic": list(VALID_TOPICS)})
        result = apply_topic_cleaning(df)
        assert set(result["topic"]) == VALID_TOPICS

    def test_unknown_topic_is_left_as_is(self):
        """An unrecognised topic should survive cleaning unchanged (lowercased only)."""
        df = pd.DataFrame({"topic": ["Some random topic"]})
        result = apply_topic_cleaning(df)
        assert result["topic"].iloc[0] == "some random topic"


# ── text cleaning tests ───────────────────────────────────────────────────────

class TestTextCleaning:

    def test_newlines_are_removed(self):
        df = pd.DataFrame({"text": ["line one\nline two\nline three"]})
        result = apply_text_cleaning(df)
        assert "\n" not in result["text"].iloc[0]

    def test_edit_history_watch_prefix_is_removed(self):
        df = pd.DataFrame({"text": ["edit,history,watchSome article text"]})
        result = apply_text_cleaning(df)
        assert not result["text"].iloc[0].startswith("edit,history,watch")

    def test_edit_history_watch_mid_string_is_not_removed(self):
        """The regex uses ^ so it should only strip the prefix, not mid-string occurrences."""
        df = pd.DataFrame({"text": ["Normal text edit,history,watch more text"]})
        result = apply_text_cleaning(df)
        assert "edit,history,watch" in result["text"].iloc[0]

    def test_clean_text_is_unchanged(self):
        df = pd.DataFrame({"text": ["A perfectly clean article."]})
        result = apply_text_cleaning(df)
        assert result["text"].iloc[0] == "A perfectly clean article."


# ── date handling tests ───────────────────────────────────────────────────────

class TestDateHandling:

    def test_date_column_is_parsed_to_datetime(self):
        df = pd.DataFrame({"date": ["2024-03-15", "2024-03-16"]})
        df["date"] = pd.to_datetime(df["date"])
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_sort_by_date_ascending(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-03-20", "2024-03-15", "2024-03-18"])
        })
        df = df.sort_values("date", ascending=True).reset_index(drop=True)
        assert df["date"].iloc[0] == pd.Timestamp("2024-03-15")
        assert df["date"].iloc[-1] == pd.Timestamp("2024-03-20")

    def test_incorrect_dates_are_filtered_out(self):
        current = datetime.datetime.now()
        correct_date = pd.Timestamp(f"{current.year}-{current.month}-{current.day}")
        wrong_date = pd.Timestamp("2000-01-01")

        df = pd.DataFrame({"date": [correct_date, wrong_date, correct_date]})
        df_filtered = df[df["date"] == correct_date]
        assert len(df_filtered) == 2
        assert wrong_date not in df_filtered["date"].values


# ── column dropping tests ─────────────────────────────────────────────────────

class TestColumnDropping:

    def test_headings_column_is_dropped(self):
        df = pd.DataFrame({
            "headings": ["h1", "h2"],
            "text": ["t1", "t2"],
            "topic": ["sports", "sports"],
            "date": ["2024-01-01", "2024-01-02"],
        })
        df.drop(["headings"], axis=1, inplace=True)
        assert "headings" not in df.columns

    def test_other_columns_survive_drop(self):
        df = pd.DataFrame({
            "headings": ["h1"],
            "text": ["t1"],
            "topic": ["sports"],
            "date": ["2024-01-01"],
        })
        df.drop(["headings"], axis=1, inplace=True)
        for col in ("text", "topic", "date"):
            assert col in df.columns
