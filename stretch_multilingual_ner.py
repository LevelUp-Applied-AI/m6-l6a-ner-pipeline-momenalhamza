"""
Module 6 Week A — Stretch: Multilingual NER Comparison

Professional multilingual Named Entity Recognition pipeline
for English and Arabic climate articles.

Models:
1. spaCy multilingual model:
   - xx_ent_wiki_sm

2. Hugging Face multilingual transformer:
   - Davlan/xlm-roberta-base-wikiann-ner

Author: Your Name
"""

# =========================
# IMPORTS
# =========================

from collections import Counter, defaultdict
import re
import unicodedata

import pandas as pd
import spacy

from transformers import pipeline


# =========================
# CONFIGURATION
# =========================

DATA_PATH = "data/climate_articles.csv"

TEXT_COLUMN_CANDIDATES = [
    "text",
    "content",
    "article",
    "body"
]

MIN_TEXTS_PER_LANGUAGE = 20

SUPPORTED_LANGUAGES = ["english", "arabic"]

LABEL_MAPPING = {
    "PER": "PERSON",
    "LOC": "GPE",
    "ORG": "ORG",
    "MISC": "MISC"
}


# =========================
# LOAD DATA
# =========================

def load_data(filepath=DATA_PATH):
    """
    Load climate dataset safely and validate structure.

    Args:
        filepath (str): Path to dataset.

    Returns:
        pd.DataFrame
    """

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError("Dataset is empty.")

    text_column = None

    for col in TEXT_COLUMN_CANDIDATES:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        raise ValueError(
            f"No valid text column found. "
            f"Expected one of: {TEXT_COLUMN_CANDIDATES}"
        )

    df = df[[text_column]].copy()

    df.rename(columns={text_column: "text"}, inplace=True)

    df.dropna(subset=["text"], inplace=True)

    df["text"] = df["text"].astype(str).str.strip()

    df = df[df["text"].str.len() > 0]

    df.reset_index(drop=True, inplace=True)

    print(f"\nLoaded dataset successfully.")
    print(f"Total documents: {len(df)}")

    return df


# =========================
# LANGUAGE DETECTION
# =========================

def is_arabic(text):
    """
    Detect Arabic text using Unicode ranges.

    Args:
        text (str)

    Returns:
        bool
    """

    arabic_pattern = re.compile(r'[\u0600-\u06FF]')

    return bool(arabic_pattern.search(text))


def detect_language(df):
    """
    Detect language for each document.

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """

    df["language"] = df["text"].apply(
        lambda text: "arabic" if is_arabic(text) else "english"
    )

    language_counts = df["language"].value_counts()

    print("\nLanguage Distribution:")
    print(language_counts)

    return df


# =========================
# TEXT CLEANING
# =========================

def normalize_text(text):
    """
    Lightweight normalization for multilingual text.

    Args:
        text (str)

    Returns:
        str
    """

    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================
# LOAD MODELS
# =========================

def load_spacy_model():
    """
    Load multilingual spaCy model.

    Returns:
        spacy.Language
    """

    print("\nLoading spaCy multilingual model...")

    nlp = spacy.load("xx_ent_wiki_sm")

    return nlp


def load_hf_model():
    """
    Load multilingual Hugging Face NER pipeline.

    Returns:
        transformers.pipeline
    """

    print("\nLoading Hugging Face multilingual model...")

    ner_pipeline = pipeline(
        task="ner",
        model="Davlan/xlm-roberta-base-wikiann-ner",
        aggregation_strategy="simple"
    )

    return ner_pipeline


# =========================
# SPACY NER
# =========================

def run_spacy_ner(texts, nlp):
    """
    Run spaCy multilingual NER.

    Args:
        texts (list)
        nlp (spacy.Language)

    Returns:
        list
    """

    results = []

    # nlp.pipe is MUCH faster than looping document by document
    docs = nlp.pipe(texts, batch_size=16)

    for text, doc in zip(texts, docs):

        entities = []

        for ent in doc.ents:

            label = LABEL_MAPPING.get(ent.label_, ent.label_)

            entities.append({
                "text": ent.text,
                "label": label
            })

        results.append({
            "text": text,
            "entities": entities
        })

    return results


# =========================
# HUGGING FACE NER
# =========================

def run_hf_ner(texts, hf_pipeline):
    """
    Run multilingual Hugging Face NER.

    Args:
        texts (list)
        hf_pipeline

    Returns:
        list
    """

    results = []

    for text in texts:

        predictions = hf_pipeline(text)

        entities = []

        for pred in predictions:

            label = LABEL_MAPPING.get(
                pred["entity_group"],
                pred["entity_group"]
            )

            entities.append({
                "text": pred["word"],
                "label": label
            })

        results.append({
            "text": text,
            "entities": entities
        })

    return results


# =========================
# STATISTICS
# =========================

def compute_statistics(results, language, model_name):
    """
    Compute multilingual NER statistics.

    Args:
        results (list)
        language (str)
        model_name (str)

    Returns:
        dict
    """

    total_entities = 0

    label_counter = Counter()

    example_entities = defaultdict(list)

    documents_with_no_entities = 0

    total_words = 0

    for item in results:

        text = item["text"]

        entities = item["entities"]

        words = text.split()

        total_words += len(words)

        if len(entities) == 0:
            documents_with_no_entities += 1

        total_entities += len(entities)

        for ent in entities:

            label_counter[ent["label"]] += 1

            # Keep only first 3 examples
            if len(example_entities[ent["label"]]) < 3:
                example_entities[ent["label"]].append(ent["text"])

    entity_density = (
        (total_entities / total_words) * 100
        if total_words > 0 else 0
    )

    no_entity_rate = (
        (documents_with_no_entities / len(results)) * 100
        if len(results) > 0 else 0
    )

    stats = {
        "language": language,
        "model": model_name,
        "documents": len(results),
        "total_entities": total_entities,
        "entity_density": round(entity_density, 2),
        "no_entity_rate": round(no_entity_rate, 2),
        "label_counts": dict(label_counter),
        "examples": dict(example_entities)
    }

    return stats


# =========================
# REPORT GENERATION
# =========================

def generate_report(statistics):
    """
    Generate comparison report.

    Args:
        statistics (list)

    Returns:
        pd.DataFrame
    """

    rows = []

    for stat in statistics:

        row = {
            "Language": stat["language"],
            "Model": stat["model"],
            "Documents": stat["documents"],
            "Total Entities": stat["total_entities"],
            "Entity Density": stat["entity_density"],
            "No Entity Rate (%)": stat["no_entity_rate"]
        }

        for label, count in stat["label_counts"].items():
            row[label] = count

        rows.append(row)

    report_df = pd.DataFrame(rows)

    report_df.fillna(0, inplace=True)

    report_df.to_csv(
        "multilingual_ner_comparison.csv",
        index=False
    )

    print("\nComparison Report:")
    print(report_df)

    print("\nReport saved:")
    print("multilingual_ner_comparison.csv")

    return report_df


# =========================
# MAIN
# =========================

def main():

    print("=" * 60)
    print("MULTILINGUAL NER COMPARISON PIPELINE")
    print("=" * 60)

    # -------------------------
    # Load dataset
    # -------------------------

    df = load_data()

    # -------------------------
    # Detect language
    # -------------------------

    df = detect_language(df)

    # -------------------------
    # Normalize text
    # -------------------------

    df["text"] = df["text"].apply(normalize_text)

    # -------------------------
    # Select balanced samples
    # -------------------------

    english_df = (
        df[df["language"] == "english"]
        .head(MIN_TEXTS_PER_LANGUAGE)
    )

    arabic_df = (
        df[df["language"] == "arabic"]
        .head(MIN_TEXTS_PER_LANGUAGE)
    )

    english_texts = english_df["text"].tolist()

    arabic_texts = arabic_df["text"].tolist()

    print("\nSelected Samples:")
    print(f"English texts: {len(english_texts)}")
    print(f"Arabic texts: {len(arabic_texts)}")

    # -------------------------
    # Load models
    # -------------------------

    spacy_model = load_spacy_model()

    hf_model = load_hf_model()

    all_statistics = []

    # ==================================================
    # ENGLISH — SPACY
    # ==================================================

    print("\nRunning spaCy on English texts...")

    en_spacy_results = run_spacy_ner(
        english_texts,
        spacy_model
    )

    en_spacy_stats = compute_statistics(
        en_spacy_results,
        language="english",
        model_name="spaCy_xx_ent_wiki_sm"
    )

    all_statistics.append(en_spacy_stats)

    # ==================================================
    # ARABIC — SPACY
    # ==================================================

    print("\nRunning spaCy on Arabic texts...")

    ar_spacy_results = run_spacy_ner(
        arabic_texts,
        spacy_model
    )

    ar_spacy_stats = compute_statistics(
        ar_spacy_results,
        language="arabic",
        model_name="spaCy_xx_ent_wiki_sm"
    )

    all_statistics.append(ar_spacy_stats)

    # ==================================================
    # ENGLISH — HUGGING FACE
    # ==================================================

    print("\nRunning Hugging Face on English texts...")

    en_hf_results = run_hf_ner(
        english_texts,
        hf_model
    )

    en_hf_stats = compute_statistics(
        en_hf_results,
        language="english",
        model_name="XLM_Roberta_WikiANN"
    )

    all_statistics.append(en_hf_stats)

    # ==================================================
    # ARABIC — HUGGING FACE
    # ==================================================

    print("\nRunning Hugging Face on Arabic texts...")

    ar_hf_results = run_hf_ner(
        arabic_texts,
        hf_model
    )

    ar_hf_stats = compute_statistics(
        ar_hf_results,
        language="arabic",
        model_name="XLM_Roberta_WikiANN"
    )

    all_statistics.append(ar_hf_stats)

    # -------------------------
    # Generate report
    # -------------------------

    generate_report(all_statistics)

    print("\nPipeline completed successfully.")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    main()