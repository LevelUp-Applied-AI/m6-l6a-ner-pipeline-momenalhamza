"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import unicodedata
import pandas as pd
import numpy as np
import spacy
from transformers import pipeline as hf_pipeline


def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset."""
    return pd.read_csv(filepath)


def explore_data(df):
    """Summarize basic corpus statistics."""
    word_counts = df["text"].astype(str).str.split().str.len()
    return {
        "shape": tuple(df.shape),
        "lang_counts": df["language"].value_counts().to_dict(),
        "category_counts": df["category"].value_counts().to_dict(),
        "text_length_stats": {
            "mean": float(word_counts.mean()),
            "min": int(word_counts.min()),
            "max": int(word_counts.max()),
        },
    }


def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis."""
    normalized = unicodedata.normalize("NFC", str(text))
    doc = nlp(normalized)
    tokens = []
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        lemma = tok.lemma_.strip().lower()
        if lemma:
            tokens.append(lemma)
    return tokens


def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER."""
    en_df = df[df["language"] == "en"]
    rows = []
    for _, row in en_df.iterrows():
        doc = nlp(str(row["text"]))
        for ent in doc.ents:
            rows.append({
                "text_id": row["id"],
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })
    return pd.DataFrame(
        rows,
        columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"],
    )


def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER."""
    en_df = df[df["language"] == "en"]
    rows = []
    for _, row in en_df.iterrows():
        text = str(row["text"])
        preds = ner_pipeline(text)

        # Merge subword tokens (## prefix) and consecutive tokens of the same entity
        merged = []
        for p in preds:
            raw_label = p["entity"]
            label = raw_label.split("-", 1)[1] if "-" in raw_label else raw_label
            word = p["word"]
            start = int(p["start"])
            end = int(p["end"])

            is_continuation = (
                merged
                and (word.startswith("##") or raw_label.startswith("I-"))
                and merged[-1]["entity_label"] == label
                and start <= merged[-1]["end_char"] + 1
            )
            if is_continuation:
                last = merged[-1]
                last["end_char"] = end
                last["entity_text"] = text[last["start_char"]:end]
            else:
                merged.append({
                    "text_id": row["id"],
                    "entity_text": text[start:end],
                    "entity_label": label,
                    "start_char": start,
                    "end_char": end,
                })
        rows.extend(merged)
    return pd.DataFrame(
        rows,
        columns=["text_id", "entity_text", "entity_label", "start_char", "end_char"],
    )


def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face."""
    spacy_keys = {(int(r.text_id), r.entity_text) for r in spacy_df.itertuples()}
    hf_keys = {(int(r.text_id), r.entity_text) for r in hf_df.itertuples()}

    return {
        "spacy_counts": spacy_df["entity_label"].value_counts().to_dict(),
        "hf_counts": hf_df["entity_label"].value_counts().to_dict(),
        "total_spacy": len(spacy_df),
        "total_hf": len(hf_df),
        "both": spacy_keys & hf_keys,
        "spacy_only": spacy_keys - hf_keys,
        "hf_only": hf_keys - spacy_keys,
    }


def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations."""
    pred_keys = {
        (int(r.text_id), r.entity_text, r.entity_label)
        for r in predicted_df.itertuples()
    }
    gold_keys = {
        (int(r.text_id), r.entity_text, r.entity_label)
        for r in gold_df.itertuples()
    }

    tp = len(pred_keys & gold_keys)
    precision = tp / len(pred_keys) if pred_keys else 0.0
    recall = tp / len(gold_keys) if gold_keys else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    # Load spaCy and HF models once, reuse across functions
    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")

    # Load and explore
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape: {summary['shape']}")
            print(f"Languages: {summary['lang_counts']}")
            print(f"Categories: {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")

        # Preprocess a sample to verify your function
        sample_row = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

        # spaCy NER across the English corpus
        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")

        # HF NER across the English corpus
        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")

        # Compare the two systems
        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only: {len(comparison['spacy_only'])}")
                print(f"HF-only: {len(comparison['hf_only'])}")

        # Evaluate against gold standard
        gold = pd.read_csv("data/gold_entities.csv")
        if spacy_entities is not None:
            metrics = evaluate_ner(spacy_entities, gold)
            if metrics is not None:
                print(f"\nspaCy evaluation: {metrics}")
