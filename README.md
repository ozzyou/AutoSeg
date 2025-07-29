# AutoSeg: Auto-Vocabulary Semantic Segmentation
**[Atlas Lab, University of Amsterdam & TomTom](https://www.icai.ai/labs/atlas-lab)**

[Osman Ülger](ozzyou.github.io), [Maksymilian Kulicki](https://ideas-ncbr.pl/en/osoby/maksymilian-kulicki/), [Yuki Asano](https://yukimasano.github.io/), [Martin R. Oswald](https://oswaldm.github.io/)

[[`Paper`](https://arxiv.org/abs/2312.04539)] [[`Project`](https://github.com/ozzyou/autoseg.github.io)]

**Auto-Vocabulary Semantic Segmentation (AVS)** advances open-ended image understanding by eliminating the necessity to predefine object categories for segmentation (such as in Open-Vocabulary Segmentation (OVS)). Our framework, AutoSeg, autonomously identifies relevant class names using semantically enhanced BLIP embeddings, constructs a target vocabulary and segments the classes afterwards. Given that open-ended object category predictions cannot be directly compared with a fixed ground truth, we developed an LLM-based Auto-Vocabulary Evaluator (LAVE) to efficiently evaluate the automatically generated classes and their corresponding segments.

<p align="center">
  <img src="img/teaser.png" width="100%" height="100%">
</p>

## Code Release
- [x] [LLM-Based Auto-Vocabulary Evaluator (LAVE)](https://github.com/ozzyou/AutoSeg/tree/main/LAVE) to map open-ended vocabulary to target vocabulary
- [ ] BBoost for generating the auto-vocabulary (coming soon upon publication)
- [ ] Full model (coming soon upon publication)

<p align="center">
  <img src="img/bboost_pipeline.png" width="100%" height="100%">
</p>

## Cite
```
@misc{ülger2025autovocabularysemanticsegmentation,
      title={Auto-Vocabulary Semantic Segmentation}, 
      author={Osman Ülger and Maksymilian Kulicki and Yuki Asano and Martin R. Oswald},
      year={2025},
      eprint={2312.04539},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2312.04539}, 
}
```
