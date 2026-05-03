# Multilingual NER Comparison Analysis

## Overview

This project compares multilingual Named Entity Recognition (NER) performance across English and Arabic climate-related articles using two multilingual NLP models:

1. spaCy multilingual model:
   - `xx_ent_wiki_sm`

2. Hugging Face transformer model:
   - `Davlan/xlm-roberta-base-wikiann-ner`

The objective was to evaluate how multilingual NER systems behave across languages in a bilingual MENA-region context where both Arabic and English documents are commonly used.

---

# Dataset Summary

The climate dataset contained a total of 200 articles.

| Language | Documents |
|---|---|
| English | 132 |
| Arabic | 68 |

For this comparison, 20 English texts and 20 Arabic texts were selected.

---

# Models Used

## 1. spaCy Multilingual Model

Model:
`xx_ent_wiki_sm`

Characteristics:
- Lightweight
- Fast inference
- Wikipedia-trained multilingual NER model
- Uses entity labels:
  - PER
  - LOC
  - ORG
  - MISC

---

## 2. Hugging Face Transformer Model

Model:
`Davlan/xlm-roberta-base-wikiann-ner`

Characteristics:
- Transformer-based multilingual model
- Context-aware deep learning architecture
- Better semantic understanding
- Stronger multilingual performance, especially for Arabic text

---

# Label Mapping Strategy

The multilingual labels were mapped into a unified schema for easier comparison.

| Original Label | Mapped Label |
|---|---|
| PER | PERSON |
| LOC | GPE |
| ORG | ORG |
| MISC | MISC |

This mapping allowed direct comparison between the two multilingual systems.

---

# Comparison Results

| Language | Model | Documents | Total Entities | Entity Density | No Entity Rate (%) | PERSON | ORG | GPE | MISC |
|---|---|---|---|---|---|---|---|---|---|
| English | spaCy_xx_ent_wiki_sm | 20 | 93 | 7.62 | 0.0 | 12 | 39 | 27 | 15 |
| Arabic | spaCy_xx_ent_wiki_sm | 20 | 20 | 2.15 | 25.0 | 9 | 2 | 2 | 7 |
| English | XLM_Roberta_WikiANN | 20 | 93 | 7.62 | 0.0 | 10 | 55 | 28 | 0 |
| Arabic | XLM_Roberta_WikiANN | 20 | 64 | 6.88 | 0.0 | 2 | 33 | 29 | 0 |

---

# Observations

## English Performance

Both models performed strongly on English texts.

Key observations:
- Both models extracted the same total number of entities:
  - 93 entities
- Entity density was identical:
  - 7.62 entities per 100 words
- No English document had zero detected entities.

The transformer model identified more organization entities (ORG) than spaCy:
- spaCy:
  - 39 ORG entities
- XLM-RoBERTa:
  - 55 ORG entities

This suggests that the transformer model has stronger contextual understanding for organizational names in English climate articles.

---

## Arabic Performance

Arabic results showed a major difference between the two models.

### spaCy Arabic Results

spaCy detected:
- Only 20 total entities
- Very low entity density:
  - 2.15
- 25% of Arabic documents had no detected entities at all

The model particularly struggled with:
- Arabic organization names
- Geopolitical entities
- Long Arabic entity boundaries

Examples of likely missed entities:
- وزارة البيئة
- الأمم المتحدة
- اتفاقية باريس

spaCy identified:
- Only 2 ORG entities
- Only 2 GPE entities

This indicates weak Arabic multilingual support in the lightweight spaCy model.

---

### Hugging Face Arabic Results

The transformer model performed significantly better on Arabic texts.

Key improvements:
- 64 total entities detected
- Entity density increased to:
  - 6.88
- Zero Arabic documents without entities

The model successfully detected many Arabic:
- Organizations
- Countries
- Cities
- Climate-related institutions

Detected entities included examples such as:
- الأردن
- دبي
- الأمم المتحدة

The transformer architecture handled Arabic context much better than spaCy.

---

# Analysis

## Why Arabic NER Is More Difficult

Arabic Named Entity Recognition is significantly more challenging than English NER for several linguistic reasons.

### 1. No Capital Letters

English entity detection benefits from capitalization:

```text
Jordan
United Nations
```

Arabic lacks this feature:

```text
الأردن
الأمم المتحدة
```

This removes an important signal for entity identification.

---

### 2. Morphological Complexity

Arabic is morphologically rich.

Words may contain:
- prefixes
- suffixes
- conjunctions
- attached pronouns

Example:

```text
وبالأردن
```

This increases tokenization difficulty and makes entity boundaries harder to detect.

---

### 3. Flexible Entity Structure

Arabic organization names are often long and syntactically variable.

Example:

```text
وزارة البيئة الأردنية
```

Lightweight multilingual models may fail to determine:
- where the entity begins
- where the entity ends

---

### 4. Limited Arabic Training Coverage

The spaCy multilingual model appears to have weaker Arabic representation compared to English.

The transformer model, however, benefits from:
- deep contextual embeddings
- large multilingual pretraining
- stronger semantic understanding

This explains its substantially better Arabic performance.

---

# Implications for NLP in the MENA Region

The results demonstrate that multilingual NLP systems for the MENA region require strong Arabic language support.

In bilingual environments such as Jordan and the broader Middle East, NLP systems must process:
- Arabic news articles
- English reports
- mixed-language documents

Weak Arabic NER performance can negatively affect:
- information extraction
- document search
- AI assistants
- government systems
- financial document analysis
- climate monitoring platforms

The comparison also shows that lightweight multilingual models may be insufficient for production systems involving Arabic text.

Transformer-based multilingual models provide:
- higher recall
- better contextual understanding
- more reliable Arabic entity extraction

For real-world bilingual NLP systems in the MENA region, transformer-based multilingual architectures are likely the more suitable choice despite higher computational cost.

---

# Conclusion

This multilingual NER comparison demonstrated substantial performance differences between English and Arabic text processing.

Main findings:
- Both models performed well on English text.
- Arabic NER remained significantly more difficult.
- spaCy struggled heavily with Arabic entities.
- XLM-RoBERTa achieved much stronger Arabic multilingual performance.

The experiment highlights the importance of model selection when building bilingual NLP systems for Arabic-English environments.

Transformer-based multilingual models appear significantly more capable for professional multilingual NLP applications in the MENA region.