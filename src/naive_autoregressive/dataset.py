import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, Dataset, get_worker_info
from torchvision import models

import cadquery as cq
from cadquery import *
import matplotlib.pyplot as plt

from datrie import Trie
from no_interaction_vis import no_interact_show
from vocabularies import vocabularies

import os
from tqdm import tqdm


def render_cadquery_code(cadquery_code, rolls, elevations):
    # Create a temporary directory to store the images
    cq_namespace = {"r": None}
    exec("import cadquery as cq;" + cadquery_code, cq_namespace)
    images = []
    for img_idx, (roll, elevation) in enumerate(zip(rolls, elevations)):
        img = no_interact_show(
            cq_namespace["r"],
            roll=roll,
            elevation=elevation,
            interact=False,
        )
        images.append(img / 256)

    return images


class Tokenizer:
    def __init__(self, vocabularies):
        self._vocabulary = ["<PAD>", "<SOS>", "<EOS>"] + vocabularies
        self._token_to_index = {token: i for i, token in enumerate(self._vocabulary)}
        self._index_to_token = {i: token for i, token in enumerate(self._vocabulary)}
        self._pad_token = self._token_to_index["<PAD>"]
        self._sos_token = self._token_to_index["<SOS>"]
        self._eos_token = self._token_to_index["<EOS>"]
        self._vocabulary_size = len(self._vocabulary)
        self._vocabulary_set = set(self._vocabulary)

        self._max_token_length = max(len(token) for token in self._vocabulary)
        self._max_sequence_length = 512

        self._characters = sorted(list(set("".join(self._vocabulary))))
        self._trie = Trie(self._characters)
        for token, index in self._token_to_index.items():
            self._trie[token] = index

    def tokenize(self, current_expr):
        token_indices = []
        while current_expr:
            match = self._trie.longest_prefix(current_expr)
            if match:
                token_indices.append(self._token_to_index[match])
                current_expr = current_expr[len(match) :]
        token_indices = [self._sos_token] + token_indices + [self._eos_token]
        return torch.Tensor(token_indices)

    def detokenize(self, token_indices):
        tokens = [
            self._index_to_token.get(token_id, self._pad_token)
            for token_id in token_indices
        ]
        return "".join(tokens)


class CADImageDataset(Dataset):
    def __init__(self, train=True):
        self.tokenizer = Tokenizer(vocabularies)
        all_data_indices = {}
        if train:
            text_dataset_dir = "/data/nzxyin/treecad/cad-recode-v1.5/train/"
            img_dataset_dir = "/data/nzxyin/treecad/cad-recode-render-v1/cs280-cads/train"
            all_img_directories = os.listdir(img_dataset_dir)
            all_img_dir_prefixes = set(
                [
                    file_name.split("_")[0]
                    for file_name in all_img_directories
                    if file_name.endswith(".npy")
                ]
            )
            all_img_dir_prefixes = list(all_img_dir_prefixes)
            for in_list_idx in tqdm(
                range(len(all_img_dir_prefixes)),
                desc="Processing images",
                leave=False,
            ):
                file_idx = all_img_dir_prefixes[in_list_idx]
                indices_path = os.path.join(
                    img_dataset_dir, f"{file_idx}_cad_indices.npy"
                )
                with open(indices_path, "rb") as fp:
                    cad_indices = np.load(fp)
                for in_file_index, cad_index in enumerate(cad_indices):
                    batch_idx = cad_index // 10000
                    if batch_idx < 10:
                        batch_idx = f"0{batch_idx}"

                    all_data_indices[cad_index] = {
                        "img_path": os.path.join(
                            img_dataset_dir, f"{file_idx}_images.npy"
                        ),
                        "rolls_path": os.path.join(
                            img_dataset_dir, f"{file_idx}_rolls.npy"
                        ),
                        "elevations_path": os.path.join(
                            img_dataset_dir, f"{file_idx}_elevations.npy"
                        ),
                        "img_idx_in_file": in_file_index,
                        "text_path": os.path.join(
                            text_dataset_dir,
                            f"batch_{batch_idx}/{cad_index}.py",
                        ),
                    }
        else:
            text_dataset_dir = "/data/nzxyin/treecad/cad-recode-v1.5/val/"
            img_dataset_dir = "/data/nzxyin/treecad/cad-recode-render-v1/cs280-cads/val"
            all_img_directories = os.listdir(img_dataset_dir)
            all_img_dir_prefixes = set(
                [
                    file_name.split("_")[0]
                    for file_name in all_img_directories
                    if file_name.endswith(".npy")
                ]
            )
            all_img_dir_prefixes = list(all_img_dir_prefixes)
            for in_list_idx in tqdm(
                range(len(all_img_dir_prefixes)),
                desc="Processing images",
                leave=False,
            ):
                file_idx = all_img_dir_prefixes[in_list_idx]
                indices_path = os.path.join(
                    img_dataset_dir, f"{file_idx}_cad_indices.npy"
                )
                with open(indices_path, "rb") as fp:
                    cad_indices = np.load(fp)
                for in_file_index, cad_index in enumerate(cad_indices):
                    # batch_idx = cad_index // 10000
                    # if batch_idx < 10:
                    #     batch_idx = f"0{batch_idx}"

                    all_data_indices[cad_index] = {
                        "img_path": os.path.join(
                            img_dataset_dir, f"{file_idx}_images.npy"
                        ),
                        "rolls_path": os.path.join(
                            img_dataset_dir, f"{file_idx}_rolls.npy"
                        ),
                        "elevations_path": os.path.join(
                            img_dataset_dir, f"{file_idx}_elevations.npy"
                        ),
                        "img_idx_in_file": in_file_index,
                        "text_path": os.path.join(
                            text_dataset_dir,
                            f"{cad_index}.py",
                        ),
                    }
        self.all_data_indices = all_data_indices
        self.all_data_indices_key = list(all_data_indices.keys())
        self.train = train

    def __getitem__(self, index):
        cad_index = self.all_data_indices_key[index]
        data = self.all_data_indices[cad_index]
        img_path = data["img_path"]
        rolls_path = data["rolls_path"]
        elevations_path = data["elevations_path"]
        text_path = data["text_path"]
        img_idx_in_file = data["img_idx_in_file"]

        images = torch.tensor(np.load(img_path, mmap_mode="r")[img_idx_in_file])
        rolls = torch.tensor(np.load(rolls_path, mmap_mode="r")[img_idx_in_file])
        elevations = torch.tensor(
            np.load(elevations_path, mmap_mode="r")[img_idx_in_file]
        )
        with open(text_path, "r") as fp:
            cad_code = "".join([s.replace("\n", ";") for s in fp.readlines()[1:]])
            tokenized_cad_code = self.tokenizer.tokenize(cad_code)
        return (images, rolls, elevations, tokenized_cad_code)

    def __len__(self):
        if self.train:
            return 64 * 1
        else:
            return 64
        # return len(self.all_data_indices_key)


def collate_fn(batch):
    images, rolls, elevations, tokenized_cad_code = zip(*batch)
    images = torch.stack(images) / 255
    rolls = torch.stack(rolls)
    elevations = torch.stack(elevations)
    decoder_inputs = [code[:-1] for code in tokenized_cad_code]
    targets = [code[1:] for code in tokenized_cad_code]
    decoder_inputs = nn.utils.rnn.pad_sequence(
        decoder_inputs, batch_first=True, padding_value=0
    ).long()
    targets = nn.utils.rnn.pad_sequence(
        targets, batch_first=True, padding_value=0
    ).long()  # Use -100 for ignored loss
    attention_masks = decoder_inputs == 0
    for i, mask in enumerate(attention_masks):
        if not mask.any():
            attention_masks[i, 0] = True
        if mask.all():
            attention_masks[i, 0] = False

    return {
        "renderings": images.float(),
        "rolls": rolls.float(),
        "elevations": elevations.float(),
        "decoder_inputs": decoder_inputs,
        "targets": targets,
        "attention_masks": attention_masks,
    }
