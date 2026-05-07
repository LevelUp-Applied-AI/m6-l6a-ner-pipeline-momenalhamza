# Stretch 6B-S2 — Cross-Lingual Embedding Comparison

**Model:** `bert-base-multilingual-cased` (mean-pool over last hidden states, attention-masked)
**Dataset:** 10 English + 10 Arabic climate texts from `data/climate_articles.csv`, paired by topic across all four categories (policy, science, impact, adaptation)
**Artifacts:** `output/cross_lingual_heatmap.png`, `output/cross_lingual_similarity_matrix.csv`, `output/cross_lingual_pair_scores.csv`, `output/cross_lingual_summary.json`

## Block-level similarity

| Block                                | Mean cosine |
|--------------------------------------|-------------|
| Within English (off-diagonal)        | 0.7164      |
| Within Arabic (off-diagonal)         | 0.7966      |
| Cross-lingual (all 100 EN–AR pairs)  | 0.5995      |
| **Same-topic cross-lingual** (10)    | **0.6915**  |
| Different-topic cross-lingual (90)   | 0.5893      |
| Lift (same-topic − different-topic)  | **+0.1022** |

| Retrieval direction | Top-1 accuracy | Mean rank |
|---------------------|----------------|-----------|
| English query → Arabic gallery | **0.90** | 1.10 |
| Arabic query → English gallery | **0.60** | 1.50 |

## (a) How well does multilingual BERT capture cross-lingual similarity?

The multilingual model demonstrably places paired English/Arabic climate texts in the same neighbourhood of its embedding space, but cross-lingual similarity sits a clear notch below within-language similarity. Same-topic English/Arabic pairs average **0.6915** cosine, while within-English and within-Arabic off-diagonal averages are **0.7164** and **0.7966** — so simply switching languages costs ~0.10 of similarity even when the underlying topic is identical. The signal is still useful: every same-topic pair scored above the all-cross average of 0.5995, and same-topic cross pairs averaged **+0.1022 above different-topic cross pairs**, which is more than enough lift to drive correct retrieval. EN→AR retrieval was correct for 9 of 10 queries (mean rank 1.10), and the strongest pairs — *Mafraq solar 200 MW* at **0.7595**, *IPCC AR6* at **0.7375**, and *NASA 2023 warmest year* at **0.7282** — all benefit from heavy shared named-entity content (proper nouns, numerals, units) that mBERT carries through its shared WordPiece vocabulary largely unchanged. The two failures are instructive: the English IPCC text (id=1) was closer to a *different* Arabic policy text (cosine 0.7516) than to its own Arabic translation (id=79, 0.7375), and the Arabic Great Barrier Reef text (id=89) ranked 3rd against its English counterpart. Both failures are policy/science texts that share heavy boilerplate (*"تغير المناخ"*, *"الهيئة الحكومية الدولية"*, *"climate change"*, *"IPCC"*) with neighbours in the same category — the model is genuinely capturing cross-lingual *topic* structure but is also pulled toward category-level generic vocabulary, and the directional asymmetry (AR→EN top-1 only 0.60) suggests the Arabic side of the embedding manifold is more compressed (within-AR mean 0.7966 vs within-EN 0.7164), so Arabic queries have a harder time discriminating among English candidates.

## (b) Implications for bilingual NLP tools in the MENA region

For practical bilingual deployments — climate-policy search across Arabic and English archives, cross-lingual document classification, IPCC/COP report retrieval — these results say a single multilingual model is a viable foundation but not a free lunch. A 90% top-1 EN→AR retrieval rate on topic-paired climate text is strong enough to power features like "find the Arabic version of this English brief" or to bootstrap a bilingual classifier from English-only labels (zero-shot transfer), and the ~0.10 same-vs-different-topic lift is large enough that simple cosine thresholding will separate on-topic Arabic results from off-topic ones. However, the AR→EN asymmetry (60% top-1) is a real deployment risk: an Arabic-first product in Amman or Riyadh that uses the same embeddings to retrieve supporting English literature will return the wrong document roughly four times in ten on near-translations, so production systems should (1) use direction-aware evaluation rather than averaging both ways, (2) layer reranking — e.g., a small multilingual cross-encoder over the top-k mBERT candidates — for any query path that originates in Arabic, and (3) consider language-specific normalisation of high-frequency category boilerplate (`تغير المناخ`, `IPCC`, `COP`) since those tokens are what cause genuine pairs like IPCC AR6 to lose to neighbours in the same category. The encouraging finding is that the multilingual space *is* aligned enough that same-topic Jordanian content like *Disi Water Conveyance* (cross-sim 0.7057) and *Mafraq solar 200 MW* (0.7595) cluster cleanly across languages — so for MENA-specific applications grounded in concrete projects, place names, and institutions, mBERT off the shelf is already useful, and the remaining error budget can be closed with reranking rather than by training separate per-language pipelines.
