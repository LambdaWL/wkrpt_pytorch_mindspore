import os
import json
import random

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


# NOTE: Defining Mindspore dataset does not require inheritence from predefined classes.
class IEMOCAP:
    def __init__(self, data_dir, split):
        """
        Args:
            data_dir: Directory for storing datasets and vocabulary files
            split: one of ["train", "test"]
        """
        data_path = os.path.join(data_dir, f"IEMOCAP_{split}.json")
        label_vocab_path = os.path.join(data_dir, "IEMOCAP_label_vocab.pkl")

        with open(data_path, encoding="utf-8") as f:
            raw_dialogues = json.load(f)
        with open(label_vocab_path, "rb") as f:
            label_vocab = pickle.load(f)
        
        self.dialogues = [
            {
                "labels": [
                    label_vocab["stoi"][u["label"]] if "label" in u.keys() else -1
                    for u in dialogue
                ],
                "features": [u["feature"] for u in dialogue], 
                "length": len(dialogue),
            }
            for dialogue in raw_dialogues
        ]
        self.num_class = len(label_vocab["stoi"])
        self.feature_dim = len(self.dialogues[0]["features"][0])

        random.shuffle(self.dialogues)
        print(f"IEMOCAP/{split}: {len(self.dialogues)} dialogues loaded.")
    
    def pad_dialogues(self, label_padding=-1, feature_padding=0.0):
        """ Ensure that all dialogues have the same number of labels/features. """
        feature_dim = self.feature_dim
        max_length = max([dialogue["length"] for dialogue in self.dialogues])

        for dialogue in self.dialogues:
            pad_length = max_length - dialogue["length"]
            dialogue["labels"] += [label_padding] * pad_length
            dialogue["features"] += [[feature_padding] * feature_dim] * pad_length
    
    def to_numpy(self):
        """ Convert all features and labels into numpy format (fp32 and int32 precisions). """
        for dialogue in self.dialogues:
            dialogue["labels"] = np.array(dialogue["labels"], np.int32)
            dialogue["features"] = np.array(dialogue["features"], np.float32)
    
    def __len__(self):
        return len(self.dialogues)
    
    def __getitem__(self, index):
        return (
            self.dialogues[index]["features"],
            self.dialogues[index]["labels"]
        )

            
class TorchIEMOCAP(TorchDataset):
    def __init__(self, dataset: IEMOCAP):
        self.data = dataset
    
    def __getitem__(self, index):
        """ 
        Return:
            features: Tensor of shape (qlen, D)
            labels: Tensor of shape (qlen,)
        """
        features, labels = self.data[index]
        return (
            torch.Tensor(features),
            torch.LongTensor(labels),
        )
    
    def __len__(self):
        return len(self.data)
    
