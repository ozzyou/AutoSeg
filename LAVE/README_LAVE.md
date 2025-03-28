# LAVE: LLM-based Auto-Vocabulary Evaluator
This repository contains the code for the **L**LM-based **A**uto-**V**ocabulary **E**valuator from the paper [Auto-Vocabulary Semantic Segmentation](https://arxiv.org/abs/2312.04539) by Osman Ülger, Maksymilian Kulicki, Yuki Asano and Martin R. Oswald.

## Instructions
Download [Llama-2-7b](https://www.llama.com/llama-downloads/) (or any other preferred LLM) and put it under `LAVE/llama`.

Create a virtual environment for the project and install the requirements
```
conda create --name lave python=3.8
source activate lave
pip install -r requirements_lave.txt
```

Add the vocabulary that you want to map to to `constants.py` and update the `DATASET_CATALOG`.

To map, run
```
torchrun --nproc_per_node=1 lave.py <JSON file with auto-vocbulary as list of classes> \
                                    <folder with your predicted auto-vocbulary masks> \
                                    <folder to save updated masks> \
                                    <dataset name as mentioned in constants.py> \
                                    --<optional arguments> <optional argument value>
```

If your vocabulary is large, consider using a small `--llm_batch_size` to prevent memory issues. Consider checking out the other parameters you can pass to the function, such as a list of classes to always ignore (`--hard_ignore_classes`).

## Notes
* We are using Python 3.8, PyTorch 1.13 for CUDA 11.6 with one NVIDIA GeForce RTX 3090.
* The accuracy of the mapping might fluctuate depending on the size of the target vocabulary as the search space expands. For large target vocabularies, we recommend using an LLM with more parameters such as Llama-2-70b.

## Cite
If useful to you, please cite our paper.
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
