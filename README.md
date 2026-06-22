# Multilingual Health Question Answering in Low-Resource African Languages

**Machine Learning Techniques I — Final Course Project**
Zindi username: `Winston_Russel_Nji`

**Best public leaderboard score: 0.514958**
(ROUGE-1: 0.5019 | ROUGE-L: 0.4275 | LLM Judge: 0.658)

[Demo Video](https://drive.google.com/file/d/1Y9k_auxdv1uDKCyzvXyx3tcqcci0iq3K/view?usp=sharing) | [Full Report](https://docs.google.com/document/d/1ZKUJSBk-u1K-nlJEg0jNN6I_BsPxMoryMhd6jc0AVMk/edit?usp=sharing) | [Colab Notebook](https://colab.research.google.com/drive/1TRKt3dYSMvV4GcX9JvSoJC-ZsAB4wmCT?usp=sharing)

---

## Overview

This project addresses the [Zindi Multilingual Health QA challenge](https://zindi.africa), which requires systems to answer health questions in the same language they were asked, across nine language-country configurations spanning Akan, Amharic, Luganda, Swahili, and English.

Submissions are scored on three metrics: ROUGE-1 F1, ROUGE-L F1, and LLM-as-a-Judge. The central finding of the project was unexpected: **retrieval beat generation on every meaningful metric**. No generative model tested, including fine-tuned mT5, Flan-T5, and NLLB-200, outperformed simply returning the nearest training answer. The final system uses no neural model at inference time at all.

---

## Dataset

| Subset | Language | Country | Training Examples | % of Total |
|---|---|---|---|---|
| Eng_Uga | English | Uganda | 7,624 | 25.6% |
| Aka_Gha | Akan | Ghana | 4,455 | 14.9% |
| Eng_Gha | English | Ghana | 4,443 | 14.9% |
| Eng_Eth | English | Ethiopia | 3,915 | 13.1% |
| Lug_Uga | Luganda | Uganda | 3,383 | 11.3% |
| Eng_Ken | English | Kenya | 2,080 | 7.0% |
| Swa_Ken | Swahili | Kenya | 2,070 | 6.9% |
| Amh_Eth | Amharic | Ethiopia | 1,845 | 6.2% |

**Total:** 29,814 training records, 6,686 validation records, 2,618 test records.

Four of the five languages use Latin script. Amharic uses Ethiopic (Ge'ez), which has no word-boundary whitespace and falls entirely outside the ASCII range. This single difference breaks word-level tokenization for Amharic and motivates the character n-gram approach used throughout the retrieval experiments.

---

## Approach

The project explored two directions.

**Generative approach:** Fine-tune or zero-shot a multilingual seq2seq model (NLLB-200, mT5-small, Flan-T5-base) to produce health answers directly. All three models were tested with and without fine-tuning, and with and without retrieved context (RAG).

**Retrieval approach:** Index training questions using TF-IDF character n-grams or BM25, then at inference return the training answer whose question is most similar to the test question. No generation, no GPU required at inference.

Retrieval won decisively. The best generative result (mT5-small RAG, ROUGE-1 0.2490) was roughly 70% lower than pure TF-IDF retrieval (0.4233). The second half of the project focused entirely on improving the retrieval mechanism.

---

## Experiments

| Exp ID | Name | Val ROUGE-1 | Val ROUGE-L | Leaderboard |
|---|---|---|---|---|
| 1a | TF-IDF Retrieval Baseline | 0.4207 | 0.3655 | |
| 1b | NLLB-200 Zero-Shot | 0.0074 | 0.0071 | |
| 1c | NLLB-200 Fine-Tuned | 0.3049 | 0.2366 | |
| 2 | mT5-Small Prompted Fine-Tune | 0.0007 | 0.0006 | |
| 3 | mT5-Small Fixed Generation | 0.1597 | 0.1317 | |
| 4 | mT5-Small RAG | 0.2490 | 0.1868 | |
| 5 | Pure TF-IDF Retrieval | 0.4233 | 0.3496 | 0.4435 |
| 6 | Top-3 Retrieval, Longest Answer | 0.3829 | 0.2994 | |
| 7 | Language-Specific TF-IDF | 0.4251 | 0.3500 | 0.4523 |
| 8 | Flan-T5-Base Zero-Shot | 0.0783 | 0.0665 | |
| 9 | Flan-T5-Base RAG | 0.1082 | 0.0969 | |
| 10 | Per-Language Strategy Ensemble | 0.2464 | 0.2013 | |
| 11 | BM25 Retrieval | 0.4041 | 0.3495 | 0.4816 |
| 12 | BM25 Per-Language | 0.4046 | 0.3486 | 0.4798 |
| 13 | TF-IDF + BM25 Hard Switch | 0.4041 | 0.3495 | 0.4798 |
| 14 | Interpolation alpha=0.5 | 0.4181 | 0.3639 | 0.5012 |
| 15 | Interpolation alpha=0.3 | 0.4135 | 0.3588 | |
| 15b | Interpolation alpha=0.7 | 0.4249 | 0.3710 | 0.5095 |
| **15c** | **Interpolation alpha=0.9** | **0.4277** | **0.3741** | **0.5150** |

---

## Final Method

The winning approach uses weighted score interpolation between TF-IDF character n-grams and BM25 word tokens.

For each test question, both retrievers independently score every training example. Scores are normalized per query to a 0 to 1 range, then combined as:

`combined_score = alpha * tfidf_similarity + (1 - alpha) * bm25_normalized`

The training example with the highest combined score is returned as the answer. At alpha=0.9, TF-IDF character n-grams carry 90% of the retrieval decision. No generation step is applied.

**Why character n-grams?** The `char_wb` analyzer with ngram range (3, 5) operates on raw character sequences, making it script-agnostic. A trigram like `alt` appears in both `health` and `healthcare` regardless of surrounding context, and Ethiopic character sequences repeat across related health questions in the same way.

**Why BM25 at 10%?** BM25 adds term frequency saturation and document length normalization on top of TF-IDF. Pure TF-IDF scored 0.4523 on the leaderboard; the interpolation ensemble at alpha=0.9 scored 0.5150. The BM25 component still provides a small but measurable improvement even at minimal weight.

---

## Key Findings

**Retrieval beats generation on this task.** ROUGE rewards token overlap with reference answers. Training answers share vocabulary and phrasing with related test answers, which makes retrieved answers near-ideal candidates by construction. Generated answers can be factually correct but use different phrasing, and ROUGE penalizes that.

**The interpolation required proper per-query normalization.** Experiment 13 tried a winner-takes-all ensemble and degenerated to pure BM25 because BM25 raw scores dominated TF-IDF cosine similarities after naive normalization. Experiment 14 fixed this by normalizing both scores per query independently before combining them, which let both systems contribute to every decision.

**TF-IDF character n-grams are the dominant signal.** Higher alpha (more TF-IDF weight) improved scores at every step from 0.3 through 0.5, 0.7, and 0.9. BM25 is providing a correction term, not a co-equal signal.

**Amharic never exceeded 0.03 ROUGE-1 in any experiment.** Ethiopic script falls entirely outside the Latin character n-gram vocabulary. With only 1,845 training examples, the retrieval pool is also too small to reliably find close matches. This is a data and script problem, not a tuning problem.

---

## Leaderboard Progression

All 10 submitted results improved monotonically after the first two format-error submissions were resolved.

| Submission | Experiment | ROUGE-1 | ROUGE-L | LLM Judge | Overall |
|---|---|---|---|---|---|
| 1 | Exp 7 (wrong format) | 0.0000 | 0.0000 | 0.0026 | 0.0007 |
| 2 | Exp 7 (missing column) | 0.4351 | 0.3514 | 0.0000 | 0.2910 |
| 3 | Exp 5 Pure TF-IDF | 0.4322 | 0.3533 | 0.5878 | 0.4435 |
| 4 | Exp 7 Lang-Specific TF-IDF | 0.4351 | 0.3514 | 0.6204 | 0.4523 |
| 5 | Exp 11 BM25 Global | 0.4731 | 0.3977 | 0.6130 | 0.4816 |
| 6 | Exp 12 BM25 Per-Language | 0.4695 | 0.3884 | 0.6245 | 0.4798 |
| 7 | Exp 13 Hard Switch Ensemble | 0.4731 | 0.3977 | 0.6063 | 0.4798 |
| 8 | Exp 14 Interpolation 0.5 | 0.4928 | 0.4175 | 0.6322 | 0.5012 |
| 9 | Exp 15b Interpolation 0.7 | 0.4991 | 0.4247 | 0.6450 | 0.5095 |
| **10** | **Exp 15c Interpolation 0.9** | **0.5019** | **0.4275** | **0.6580** | **0.5150** |

---

## Setup and Requirements

```bash
pip install scikit-learn pandas numpy rouge-score transformers sentencepiece rank-bm25
```

Data files (`Train.csv`, `Val.csv`, `Test.csv`, `SampleSubmission.csv`) should be placed in the working directory. The notebook is designed to run on Google Colab with Google Drive mounted at `/content/drive/MyDrive/multilingual_health_qa/`.

The retrieval experiments (Experiments 5 through 15c) require no GPU. The generative experiments (Experiments 1b through 10) were run on a Tesla T4 (15.5 GB VRAM).

---

## Limitations

The system retrieves; it does not reason. It cannot produce health information that is not already present in the training data, and it has no mechanism to flag when a retrieved answer is weakly relevant. Amharic performance (peak ROUGE-1: 0.0285) is too low for any real deployment without per-output human review. The LLM Judge score of 0.658 reflects the fact that retrieved answers sometimes address the topic but not the specific question asked.

---

## AI Use Disclosure

Claude (Anthropic) assisted with generating and debugging notebook cells for Experiments 11 through 15c, explaining hyperparameter choices, and structuring this report. All experimental decisions, code execution, result interpretation, and leaderboard submissions were my own.

---

## References

- Robertson, S., and Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.
- Xue, L., et al. (2021). mT5: A massively multilingual pre-trained text-to-text transformer. *Proceedings of NAACL 2021*.
- NLLB Team. (2022). No language left behind. *arXiv:2207.04672*.
- Wei, J., et al. (2022). Finetuned language models are zero-shot learners. *ICLR 2022*.
- Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries. *Text Summarization Branches Out*.

