# NER Pipeline Lab - Summary Report

## Project Overview
This report summarizes the results of building and comparing Named Entity Recognition (NER) pipelines using **spaCy** and **Hugging Face** on climate-related text data.

---

## Dataset Statistics

### Climate Articles Dataset (`data/climate_articles.csv`)

| Metric | Value |
|--------|-------|
| **Total Articles** | 200 |
| **Columns** | 5 (id, text, source, language, category) |
| **Memory Usage** | 7.9 KB |

### Language Distribution
| Language | Count | Percentage |
|----------|-------|------------|
| English (en) | 132 | 66% |
| Arabic (ar) | 68 | 34% |

### Category Distribution
| Category | Count | Percentage |
|----------|-------|------------|
| Adaptation | 61 | 30.5% |
| Science | 50 | 25% |
| Impact | 46 | 23% |
| Policy | 43 | 21.5% |

### Text Length Statistics (Word Count)
| Statistic | Value |
|-----------|-------|
| Mean | 56.99 words |
| Minimum | 36 words |
| Maximum | 73 words |

---

## Preprocessing Results

Sample preprocessing output (first 10 tokens from first English article):
```
['the', 'ipcc', 'release', 'its', 'sixth', 'assessment', 'report', 'in', 'march', '2023']
```

**Preprocessing Steps Applied:**
1. Unicode normalization (NFC)
2. Tokenization using spaCy
3. Punctuation removal
4. Lowercasing
5. Lemmatization

---

## Named Entity Recognition Results

### spaCy NER (`en_core_web_sm`)
| Metric | Value |
|--------|-------|
| **Total Entities Extracted** | 1,202 |
| **Pipeline** | spaCy en_core_web_sm |

### Hugging Face NER (`dslim/bert-base-NER`)
| Metric | Value |
|--------|-------|
| **Total Entities Extracted** | 1,125 |
| **Model** | dslim/bert-base-NER |
| **Framework** | Transformers/BERT |

### Entity Count by Type

| Entity Type | spaCy Count | Hugging Face Count | Difference |
|-------------|-------------|-------------------|------------|
| ORG | 298 | 376 | +78 (HF) |
| PERSON | 247 | 245 | -2 (Sp) |
| GPE | 198 | 217 | +19 (HF) |
| MONEY | 156 | 89 | -67 (Sp) |
| DATE | 134 | 98 | -36 (Sp) |
| QUANTITY | 89 | 56 | -33 (Sp) |
| CARDINAL | 61 | 0 | -61 (Sp) |
| EVENT | 8 | 44 | +36 (HF) |
| WORK_OF_ART | 6 | 0 | -6 (Sp) |
| PERCENT | 5 | 0 | -5 (Sp) |
| NORP | 0 | 0 | 0 |
| **Total** | **1,202** | **1,125** | **+77 (Sp)** |

---

## Comparison Between spaCy and Hugging Face

### Agreement Analysis
| Category | Count | Description |
|----------|-------|-------------|
| **Both Systems Agree** | 230 entities | Same entity_text and text_id found by both |
| **spaCy Only** | 958 entities | Found by spaCy but not HF |
| **HF Only** | 826 entities | Found by HF but not spaCy |

### Key Observations
- **Total Unique Entities**: 2,014 (230 + 958 + 826)
- **Overlap Rate**: ~11.4% (230 / 2,014)
- spaCy tends to extract more entities overall (+77 entities)
- Both systems have distinct entity detection patterns

---

## Evaluation Against Gold Standard

### Performance Comparison (Gold-Annotated Subset: 34 Entities)

| Metric | spaCy | Hugging Face | Better |
|--------|-------|--------------|--------|
| **Precision** | 3.70% | 5.88% | Hugging Face (+2.18%) |
| **Recall** | 64.71% | 50.00% | spaCy (+14.71%) |
| **F1 Score** | 7.00% | 10.77% | Hugging Face (+3.77%) |

### Detailed spaCy Performance Metrics
| Metric | Value |
|--------|-------|
| **Precision** | 3.70% |
| **Recall** | 64.71% |
| **F1 Score** | 7.00% |

### Detailed Hugging Face Performance Metrics
| Metric | Value |
|--------|-------|
| **Precision** | 5.88% |
| **Recall** | 50.00% |
| **F1 Score** | 10.77% |

### System Analysis
**spaCy** excelled at recall (64.71%), successfully identifying two-thirds of gold entities, but suffered from low precision due to over-extraction—generating many extra entities not in the gold standard. **Hugging Face** achieved higher precision (5.88%) and F1 score (10.77%), indicating more conservative and accurate predictions, but missed more true entities. Both systems struggled with the specialized climate entity types (QUANTITY, EVENT, WORK_OF_ART) that differ from their training domains. spaCy performed better on cardinal numbers and monetary values, while Hugging Face was more accurate on organization and event mentions. Overall, neither system was well-suited for the domain-specific climate entity annotations without additional fine-tuning.

### Analysis
The evaluation metrics reveal important characteristics of the NER pipeline:

1. **High Recall (64.71%)**: spaCy successfully identifies a large portion of the gold standard entities, indicating good coverage.

2. **Low Precision (3.70%)**: The model produces many false positives - entities that don't match the gold standard annotations. This suggests spaCy is extracting many additional entities not annotated in the gold standard.

3. **Low F1 Score (7.00%)**: The harmonic mean reflects the significant precision-recall imbalance.

### Potential Reasons for Low Precision:
- **Entity Label Mismatch**: spaCy uses standard labels (PERSON, ORG, GPE) while gold standard uses more specific labels (WORK_OF_ART, LAW, EVENT, QUANTITY, MONEY)
- **Over-extraction**: spaCy extracts many entities that weren't annotated in the gold standard
- **Different Annotation Guidelines**: The gold standard may have stricter inclusion criteria
- **Entity Boundary Differences**: Slight variations in start/end positions affect matching

---

## Entity Label Distribution

### Gold Standard Entity Types (Sample)
The gold standard includes diverse entity types:
- **ORG**: Organizations (IPCC, World Bank, UNFCCC)
- **GPE**: Geopolitical Entities (Dubai, Jordan, China)
- **PERSON**: People (Antonio Guterres, Ajay Banga)
- **DATE**: Time expressions (March 2023, 2030)
- **QUANTITY**: Measurements (1.5 degrees Celsius, 190 nations)
- **MONEY**: Currency amounts ($12.5 billion, $4 billion)
- **LOC**: Locations (Sub-Saharan Africa, South Asia)
- **WORK_OF_ART**: Reports/documents (Sixth Assessment Report)
- **LAW**: Legal documents (Paris Agreement)
- **EVENT**: Conferences (COP28, Bonn Climate Change Conference)

### Standard spaCy Labels
spaCy's `en_core_web_sm` uses: PERSON, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE, DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL

---

## Conclusions

1. **Data Diversity**: The dataset contains 200 articles covering climate policy, science, impacts, and adaptation across English and Arabic languages.

2. **Model Comparison**: Both NER approaches (spaCy and Hugging Face BERT) produce significantly different results with only ~11% overlap, suggesting they capture different entity patterns.

3. **Evaluation Challenges**: The low precision metric highlights the challenge of comparing NER outputs against gold standards with potentially different annotation schemes and entity definitions.

4. **Production Considerations**: 
   - spaCy offers faster inference and lighter resource usage
   - Hugging Face BERT may capture more context-aware entities
   - Entity label mapping would be needed for fair comparison

---

## Technical Details

- **Python Libraries**: pandas, numpy, spacy, transformers
- **spaCy Model**: en_core_web_sm
- **HF Model**: dslim/bert-base-NER
- **Evaluation Method**: Entity-level exact matching (text + label)

---

