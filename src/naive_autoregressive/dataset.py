import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
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
        token_indices = (
            [self._sos_token]
            + token_indices
            + [self._eos_token]
            + [self._pad_token] * (self._max_sequence_length - len(token_indices) - 2)
        )
        return token_indices

    def detokenize(self, token_indices):
        tokens = [
            self._index_to_token.get(token_id, self._pad_token)
            for token_id in token_indices
        ]
        return "".join(tokens)


class AutoRegressiveDataset(IterableDataset):
    def __init__(self, batch_size, num_renders, train=True):
        """
        1. Set up the attributes
        2. Initialize the tokenizer
        3. Get the content of all files in cad-recode v1.5 dataset
        4. Tokenize those contents
        """
        self.batch_size = batch_size
        self.num_renders = num_renders
        self.tokenizer = Tokenizer(vocabularies)

        # Get content of all files in cad-recode v1.5 dataset
        all_cad_code = []
        all_tokenized_texts = []

        if train:
            text_dataset_dir = "/home/brandonh/cad-recode-v1.5/train/"
            batch_ids = [f"0{i}" for i in range(1)]  # + list(range(10, 100))
            for batch_id in tqdm(batch_ids, desc="Processing batches", leave=False):
                batch_dir = os.path.join(text_dataset_dir, f"batch_{batch_id}")

                # Check if the directory exists
                for _, file_name in tqdm(
                    enumerate(os.listdir(batch_dir)),
                    desc=f"Processing files in {batch_dir}",
                ):
                    file_path = os.path.join(batch_dir, file_name)
                    if os.path.isfile(file_path):
                        with open(file_path, "r") as file:
                            contents = "".join(
                                [s.replace("\n", ";") for s in file.readlines()[1:]]
                            )
                            all_cad_code.append(contents)
                            all_tokenized_texts.append(
                                self.tokenizer.tokenize(contents)
                            )

            img_dataset_dir = (
                "/home/vint/treecad-280/datasets/cad-recode-render-v1/train"
            )
            all_imgs = []
            for batch_id, zipped_imgs_name in tqdm(
                enumerate(os.listdir(img_dataset_dir)),
                desc="Processing batches",
                leave=False,
            ):
                batch_dir = os.path.join(img_dataset_dir, zipped_imgs_name)
                # Check if the directory exists
                for _, file_name in tqdm(
                    enumerate(os.listdir(batch_dir)),
                    desc=f"Processing files in {batch_dir}",
                ):
                    file_path = os.path.join(batch_dir, file_name)
                    with open(file_path, "r") as fp:
                        images = np.load(fp)  # (100, 8, 128, 128, 3)
                        all_imgs.append(images)
            all_imgs = np.concatenate(all_imgs, axis=0) / 256

        else:
            text_dataset_dir = "/home/brandonh/cad-recode-v1.5/val/"
            for _, file_name in tqdm(
                enumerate(os.listdir(text_dataset_dir)),
                desc=f"Processing batches",
                leave=False,
            ):
                file_path = os.path.join(text_dataset_dir, file_name)
                if os.path.isfile(file_path):
                    with open(file_path, "r") as file:
                        contents = "".join(
                            [s.replace("\n", ";") for s in file.readlines()[1:]]
                        )
                        all_cad_code.append(contents)
                        all_tokenized_texts.append(self.tokenizer.tokenize(contents))

            img_dataset_dir = "/home/vint/treecad-280/datasets/cad-recode-render-v1/val"
            all_imgs = []
            for batch_id, zipped_imgs_name in tqdm(
                enumerate(os.listdir(img_dataset_dir)),
                desc="Processing batches",
                leave=False,
            ):
                batch_dir = os.path.join(img_dataset_dir, zipped_imgs_name)
                # Check if the directory exists
                for _, file_name in tqdm(
                    enumerate(os.listdir(batch_dir)),
                    desc=f"Processing files in {batch_dir}",
                ):
                    file_path = os.path.join(batch_dir, file_name)
                    with open(file_path, "r") as fp:
                        images = np.load(fp)  # (100, 8, 128, 128, 3)
                        all_imgs.append(images)
            all_imgs = np.concatenate(all_imgs, axis=0) / 256

        self.all_cad_code = all_cad_code
        self.all_tokenized_texts = torch.Tensor(all_tokenized_texts)
        self.rolls_range = torch.arange(-180, 180, 18)
        self.elevations_range = torch.arange(-90, 90, 9)
        self.random_cad_indices = torch.randperm(len(self.all_tokenized_texts))
        self.current_index = 0
        self.all_imgs = torch.Tensor(all_imgs)

    def shuffle(self):
        self.random_cad_indices = torch.randperm(len(self.all_tokenized_texts))
        self.current_index = 0

    def _produce_batch(self):
        """
        Signature heavily references the formality of the tree diffusion training
        script (tree-diffusion/scripts/train.py)
        1. No-interaction rendering at randomly sampled angles
        2. Process stuff and return
        """
        random_angles = torch.randint(
            low=0,
            high=20,
            size=(self.batch_size, self.num_renders, 2),
        )
        rolls = self.rolls_range[random_angles[:, :, 0]]
        elevations = self.elevations_range[random_angles[:, :, 1]]
        random_cad_indices = self.random_cad_indices[
            self.current_index : self.current_index + self.batch_size
        ]
        self.current_index += self.batch_size

        renderings = []
        for i in range(self.batch_size):
            renderings.append(
                render_cadquery_code(
                    self.all_cad_code[random_cad_indices[i]],
                    rolls[i],
                    elevations[i],
                )
            )
        renderings = torch.Tensor(np.array(renderings))  # (B x k x 3 x 128 x 128)

        return {
            "tokenized_texts": self.all_tokenized_texts[random_cad_indices],
            "rolls": random_angles[:, :, 0],
            "elevations": random_angles[:, :, 1],
            "renderings": renderings,
        }

    def __iter__(self):
        self.shuffle()
        all_batches = []
        for i in tqdm(
            # range(0, len(self.all_tokenized_texts), self.batch_size),
            range(2),
            desc="Producing batches",
            leave=False,
            total=2,  # len(self.all_tokenized_texts) // self.batch_size,
        ):
            batch = self._produce_batch()
            for j in range(100):
                all_batches.append(batch)
        return iter(all_batches)

    def __len__(self):
        return 2000
        # return len(self.all_tokenized_texts) // self.batch_size
