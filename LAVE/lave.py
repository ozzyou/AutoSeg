# --------------------------------------------------------
# LAVE - LLM-based Auto-Vocabulary Evaluator
# Copyright (c) 2024 O. Ulger
# Licensed under The MIT License
# Modified by O. Ulger (o.ulger@uva.nl)
# --------------------------------------------------------

import os
import json
import torch
import argparse
import numpy as np

from constants import DATASET_CATALOG
from llama.generation import Llama, Dialog
from tqdm import tqdm

class LAVE:
    def __init__(self, generator, max_gen_len, temperature, top_p, dataset, background_class, background_idx,
                 hard_ignore_classes, llm_batch_size, max_parse_len, output_root):

        # Dataset variables
        assert dataset in DATASET_CATALOG, f"Dataset {dataset} not recognized!"
        self.dataset_name = dataset
        self.dataset = DATASET_CATALOG[dataset]
        if background_class and background_class not in self.dataset:
            self.dataset.insert(background_idx, background_class)
        self.dataset_class_to_idx = {n: i for i, n in enumerate(self.dataset)}
        self.dataset_string = f"{str(self.dataset)}"
        self.background_class = background_class
        self.hard_ignore_classes = hard_ignore_classes

        # Generator variables
        self.generator = generator
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p
        self.llm_batch_size = llm_batch_size

        # Misc variables
        self.max_parse_len = max_parse_len
        self.output_root = output_root

    def create_dialog(self, name):
        """Helper to generate the dialog based on the dataset name."""
        if self.dataset_name == "ADE20K-847":
            prompt = (
                f"To which class in the ADE20K-847 dataset is '{name}' exclusively most similar to? "
                f"If {name} is not similar to any class in ADE20K or if the term describes stuff instead of things, "
                f"answer with 'background'. Reply in single quotation marks with the class name that is part of ADE20K "
                f"and do not link it to any other class name which is not part of the given list or 'background'"
            )
        elif self.dataset_name == "PC-459":
            prompt = (
                f"To which class in PASCAL-Context-459 dataset is '{name}' exclusively most similar to? "
                f"If {name} is not similar to any class in the list, answer with 'background'. Reply in single quotation marks "
                f"with the class name that is part of the list and do not link it to any other class name which is not part of the given list or 'background'"
            )
        else:
            prompt = (
                f"To which class in {self.dataset_string} is '{name}' exclusively most similar to? "
                f"If {name} is not similar to any class in the list, answer with 'void'. Reply in single quotation marks "
                f"with the class name that is part of the list and do not link it to any other class name which is not part of the given list or 'background'"
            )
        return [{"role": "user", "content": prompt}]

    def llm_mapper_batched(self, all_unique_classes):
        batched_llm_dialogs = []

        # Process names in batches and create dialogues using a helper
        for i in tqdm(range(0, len(all_unique_classes), self.llm_batch_size)):
            batch_names = all_unique_classes[i:i + self.llm_batch_size]
            llm_dialogs = [self.create_dialog(name) for name in batch_names]
            results = self.generator.chat_completion(
                llm_dialogs,
                max_gen_len=self.max_gen_len,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            batched_llm_dialogs.append((results, batch_names))

        # Parse responses from all batches
        map_dict, skipped = {}, []
        valid_set = set(self.dataset + [self.background_class])
        for results, names in batched_llm_dialogs:
            for result, name in zip(results, names):
                content = result['generation']['content']

                # Try splitting on single quotes, fallback to double quotes if needed
                tokens = content.split("'")[1:] or content.split('"')[1:]
                tokens_set = set(tokens)
                common = list(tokens_set.intersection(valid_set))

                if name in self.dataset:
                    map_dict[name] = [name]
                elif not common:
                    skipped.append(name)
                else:
                    map_dict.setdefault(name, []).extend(common)

        # Post-process mapping
        for key in list(map_dict.keys()):
            if self.max_parse_len and len(map_dict[key]) > self.max_parse_len:
                map_dict[key] = [self.background_class]
            if key in self.dataset or key == self.background_class:
                map_dict[key] = key
            else:
                map_dict[key] = map_dict[key][0]  # take the first result

        # Handle any names that were skipped. We manually map 'woman' to 'person' if 'woman' is not part of the dataset.
        # This is necessary since some LLMs refuse to make this mapping, while 'woman' is a frequent class. Change
        # 'person' to desired class, e.g. 'female'/'adult'/etc if necessary.
        for name in skipped:
            if name == 'woman' and name not in self.dataset:
                map_dict[name] = 'person'
            else:
                map_dict[name] = self.background_class

        return map_dict

    def update_predictions(self, predictions, auto_vocabulary, mapper):
        orig_mapping = {}
        for n in torch.unique(predictions):
            av_class = auto_vocabulary[n]

            assert av_class in mapper, f"{av_class} has not been mapped."
            mapped_to_class = mapper[av_class].lower()

            orig_mapping[n.item()] = self.dataset_class_to_idx[mapped_to_class]

        # Step 1: Map original class indices to temporary indices based on mapper.
        # This prevents intermediate updating of class indices in our output
        temp_mapping = {k: -i for i, k in enumerate(orig_mapping)}
        for k, v in temp_mapping.items():
            predictions[predictions == k] = v

        # Step 2: Map temporary values to final values
        final_mapping = {v: orig_mapping[k] for k, v in temp_mapping.items()}
        for k, v in final_mapping.items():
            predictions[predictions == k] = v

        return predictions


def main(args):
    torch.multiprocessing.set_start_method('spawn')

    # Build the generator with the given parameters
    generator = Llama.build(
        ckpt_dir=args.llm_ckpt_dir,
        tokenizer_path=args.llm_tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.llm_batch_size,
    )

    # Create an instance of llm_mapper (LAVE) with the parsed arguments
    lave = LAVE(
        generator,
        args.max_gen_len,
        args.temperature,
        args.top_p,
        args.dataset,
        args.background_class,
        args.background_idx,
        args.hard_ignore_classes,
        args.llm_batch_size,
        args.max_parse_len,
        args.output_root
    )

    all_unique_classes = json.load(open(args.auto_vocabulary_json, "r"))

    # Map all predicted classes to the vocabulary classes of the dataset
    mapper = lave.llm_mapper_batched(all_unique_classes)

    # Iterate over all .npy and .pt files in the input directory and update the predictions with the
    # mapped vocabulary classes.
    for filename in os.listdir(args.input_root):
        input_filepath = os.path.join(args.input_root, filename)
        if filename.endswith('.npy'):
            predictions = np.load(input_filepath, allow_pickle=True)
        elif filename.endswith('.pt'):
            predictions = torch.load(input_filepath)
        else:
            continue

        updated_predictions = lave.update_predictions(predictions=predictions, auto_vocabulary=all_unique_classes, mapper=mapper)

        # Here we save the updated predictions to pass through our evaluation metric later (differs per task). You could
        # also directly apply the metric here without saving the updated predictions, but it saves you rerunning the
        # mapping entirely in case you want to make some adjustments.
        output_filepath = os.path.join(args.output_root, filename)
        if filename.endswith('.npy'):
            np.save(output_filepath, updated_predictions)
        elif filename.endswith('.pt'):
            torch.save(updated_predictions, output_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map predicted classes to dataset vocabulary using LAVE.")

    # Required positional arguments
    parser.add_argument("auto_vocabulary_json", type=str, help="JSON file containing the auto-vocabulary")
    parser.add_argument("input_root", type=str, help="Input root directory containing predictions")
    parser.add_argument("output_root", type=str, help="Output root directory")
    parser.add_argument("dataset", type=str, help="Dataset name or identifier")

    # Optional arguments with default values
    parser.add_argument("--llm_ckpt_dir", type=str, default="./llama/llama-2-7b-chat/", help="Directory with LLM checkpoints")
    parser.add_argument("--llm_tokenizer_path", type=str, default="./llama/tokenizer.model", help="Directory with LLM tokenizer")
    parser.add_argument("--max_seq_len", type=int, default=2000, help="Maximum generation length for LLM")
    parser.add_argument("--max_gen_len", type=int, default=512, help="Maximum generation length for LLM")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p value for nucleus sampling")
    parser.add_argument("--llm_batch_size", type=int, default=4, help="Batch size for LLM mapping")
    parser.add_argument("--max_parse_len", type=int, default=3, help="Maximum parse length")
    parser.add_argument("--background_class", type=str, default="background", help="Background class name")
    parser.add_argument("--background_idx", type=int, default=0, help="Background index value")
    parser.add_argument("--hard_ignore_classes", nargs='*', default=[], help="List of classes to hard ignore")

    args = parser.parse_args()
    main(args)

