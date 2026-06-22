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
