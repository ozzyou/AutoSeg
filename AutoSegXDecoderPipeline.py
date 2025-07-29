import os
import cv2
import torch
import numpy as np

from typing import Tuple, Dict, Union
from torch.utils.data import DataLoader
from infinibatch import iterators
from PIL import Image, ImageDraw, ImageFont

from X_Decoder.modeling.utils import get_class_names
from X_Decoder.utils.arguments import load_opt_command
from X_Decoder.utils.distributed import is_main_process
from X_Decoder.pipeline.utils.misc import hook_metadata, hook_switcher, hook_opt
from X_Decoder.trainer.default_trainer import DefaultTrainer
from X_Decoder.datasets import build_eval_dataloader, build_train_dataloader
from X_Decoder.utils.distributed import is_main_process
from X_Decoder.utils.visualizer import Visualizer

from BBoost.bboost import BBoost
from detectron2.data import DatasetCatalog, MetadataCatalog

def save_avs(metadata, sample, output, labels, filename, save_dir="./output"):
    image_pth = sample[0]['file_name']
    image_ori = Image.open(image_pth).convert("RGB")
    image_ori = np.asarray(image_ori)

    output = output[0]['sem_seg'].argmax(dim=0)
    labels_idx = [(p.item(), labels[p]) for p in torch.unique(output)]

    visual = Visualizer(image_ori, metadata=metadata)
    demo = visual.draw_sem_seg_avs(output.cpu(), text_labels=labels_idx)
    demo.save(os.path.join(save_dir, filename + '_avs.png'))
    print(f"Saved prediction to {os.path.join(save_dir, filename + '_avs.png')}")

class AutoSegXDecoderPipeline:
    def __init__(self, opt):
        self._opt = opt
        self.total = 0

        self.input_folder = self._opt['INPUT_DIR']
        self.save_folder = self._opt['SAVE_DIR']
        self.passes = self._opt['PASSES']
        self.min_len = self._opt['MIN_CAP_LEN']
        self.max_len = self._opt['MAX_CAP_LEN']

        self.label_generator = BBoost(device=self._opt['device'], n_passes=self._opt['PASSES'],
                                      min_length=self._opt['MIN_CAP_LEN'], max_length=self._opt['MAX_CAP_LEN'],
                                      attention_mode=self._opt['ATT_MODE'])

    def get_dataloaders(
            self, trainer: DefaultTrainer, dataset_label: str, is_evaluation: bool):
        distributed = self._opt['world_size'] > 1
        if not hasattr(self, 'valid_loader'):
            dataloaders = build_eval_dataloader(self._opt)
            self.valid_loader = dataloaders
        else:
            dataloaders = self.valid_loader
        idx = 0 if dataset_label == 'dev' else self._opt['DATASETS']['TEST'].index(dataset_label)
        dataloader = dataloaders[idx]
        return dataloader

    @staticmethod
    def forward_func(trainer, batch):
        loss = trainer.models['default'](batch)
        return loss

    def inference(self, trainer: DefaultTrainer, input_folder, save_folder):

        model = trainer.raw_models['default'].eval()
        self._opt = hook_opt(self._opt)
        torch.cuda.empty_cache()

        with torch.no_grad():

            # First set the model to an existing dataset to obtain the OVS model's metadata (based on PASCAL VOC)
            names = get_class_names(self._opt['DATASETS']['TEST'][0])
            model.model.metadata = MetadataCatalog.get(self._opt['DATASETS']['TEST'][0])
            model.model.metadata = hook_metadata(model.model.metadata, self._opt['DATASETS']['TEST'][0])

            if 'background' in names:
                model.model.sem_seg_head.num_classes = len(names) - 1

            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(names, is_eval=True)
            hook_switcher(model, self._opt['DATASETS']['TEST'][0])

            # Then switch to the input images in question
            eval_batch_gen = self.get_dataloaders(trainer, 'autoseg_inference', is_evaluation=True)

            # Batch size for inference is set to 1
            for sample in eval_batch_gen:
                file_name = os.path.splitext(os.path.basename(sample[0]['file_name']))[0]
                auto_vocabulary = self.label_generator(sample[0])# + ['background']

                bb = {}
                for i in range(len(auto_vocabulary)):
                    bb[i] = i

                model.model.metadata = model.model.metadata.set(thing_dataset_id_to_contiguous_id=bb)
                model.model.metadata = model.model.metadata.set(stuff_classes=auto_vocabulary)
                model.model.metadata = model.model.metadata.set(thing_classes=auto_vocabulary)
                if 'background' in auto_vocabulary:
                    model.model.sem_seg_head.num_classes = len(auto_vocabulary) - 1

                model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(auto_vocabulary, is_eval=True)

                # Get the prediction on just one image for classes specific to that image
                output = model(sample)

                # Reset model
                model.model.sem_seg_head.predictor.lang_encoder.reset_text_embeddings()
                model.model.sem_seg_head.num_classes = self._opt['MODEL']['ENCODER']['NUM_CLASSES']
                model.model.metadata = MetadataCatalog.get(self._opt['DATASETS']['TEST'][0])

                save_avs(model.model.metadata, sample, output, auto_vocabulary, file_name)
