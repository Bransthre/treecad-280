"""
This file contains the code for training an autoregressive model using a dataset of code snippets.
It will receive many images and output a sequence of tokens as the cadquery code.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import models

from cadquery import *
from no_interaction_vis import no_interact_show
import matplotlib.pyplot as plt

from vocabularies import vocabularies
from datrie import Trie

import os
import wandb
import tqdm

# TODO: Set up wandb logging


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

        self._trie = Trie(self._vocabulary)
        for token, index in self._token_to_index.items():
            self._trie[token] = index

    def tokenize(self, current_expr):
        token_indices = []
        while current_expr:
            match = self._trie.longest_prefix(current_expr)
            if match:
                token_indices.append(self._token_to_index[match])
                current_expr = current_expr[len(match) :]
        token_indices += [self._pad_token] * (
            self._max_sequence_length - len(token_indices)
        )
        return token_indices

    def detokenize(self, token_indices):
        tokens = [
            self._index_to_token.get(token_id, self._pad_token)
            for token_id in token_indices
        ]
        return "".join(tokens)


class BaselineCADGenerator(nn.Module):
    """
    The module naively takes in a pair of images (target rendering, current rendering)
    and outputs a sequence of tokens resembling the CAD file code.
    """

    def __init__(self, output_dim):
        super().__init__()
        self.encoder = models.VisionTransformer(
            pretrained=True,
            image_size=224,
            patch_size=16,
            num_classes=output_dim,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
        )  # I don't know if this is the right parameters but here we go.
        self.tokenizer = Tokenizer(vocabularies)
        self.token_embedding = nn.Embedding(len(self.tokenizer._vocabulary), 768)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768, nhead=8, batch_first=True),
            num_layers=6,
        )
        self.output_dim = output_dim

    def forward(self, renderings, tokenized):
        """
        A few arbitrary design decisions, which are all heavily dependent on
        (tree-diffusion/td/learning/gpt.py).

        (1) We will concatenate (meaning 3k channels) the k provided renderings.
        They should probably accompany some angular positional embedding here.
        Assume input `renderings` is B x k x C x 224 x 224
        """
        concat_img_repr = torch.cat(renderings, dim=1)  # B x kC x 224 x 224
        image_embeddings = self.encoder(concat_img_repr * 2 - 1)  # Normalize to [-1, 1]
        image_embeddings = image_embeddings.unsqueeze(1)
        decoder_input = self.token_embedding(tokenized)
        decoder_output_logits = self.decoder(decoder_input, image_embeddings)
        return decoder_output_logits


def render_cadquery_code(cadquery_code, rolls, elevations):
    """
    1. Set up tempfiles
    2. Execute the cadquery code
    3. Render the cadquery code
    4. Return the rendered images
    """
    num_renders = rolls.shape[0]
    # Create a temporary directory to store the images
    temp_dir = os.path.join(os.getcwd(), "temp_renderings")
    os.makedirs(temp_dir, exist_ok=True)

    r = None
    exec(cadquery_code)
    for img_idx, (roll, elevation) in enumerate(zip(rolls, elevations)):
        no_interact_show(
            r,
            screenshot=os.path.join(temp_dir, f"img_{img_idx}.png"),
            roll=roll,
            elevation=elevation,
            interact=False,
        )

    # Load the images from the temporary directory
    images = []
    for img_idx in range(num_renders):
        img_path = os.path.join(temp_dir, f"img_{img_idx}.png")
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            images.append(img)
        else:
            print(f"Image {img_path} does not exist.")

    # Clean up the temporary directory
    for img_idx in range(num_renders):
        img_path = os.path.join(temp_dir, f"img_{img_idx}.png")
        if os.path.exists(img_path):
            os.remove(img_path)
    os.rmdir(temp_dir)
    return images


class AutoRegressiveDataset(IterableDataset):
    def __init__(self, batch_size, num_renders):
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
        dataset_dir = "~/cad-recode-v1.5/train/"
        batch_ids = [f"0{i}" for i in range(10)] + list(range(10, 100))
        all_cad_code = []
        all_tokenized_texts = []
        for batch_id in tqdm.tqdm(batch_ids, desc="Processing batches", leave=False):
            batch_dir = os.path.join(dataset_dir, f"batch_{batch_id}")

            # Check if the directory exists
            if os.path.exists(batch_dir):
                # Iterate through all files in the batch directory
                for _, file_name in tqdm.tqdm(
                    enumerate(os.listdir(batch_dir)),
                    desc=f"Processing files in {batch_dir}",
                ):
                    file_path = os.path.join(batch_dir, file_name)
                    if os.path.isfile(file_path):
                        with open(file_path, "r") as file:
                            content = file.read()
                            all_cad_code.append(content)
                            all_tokenized_texts.append(self.tokenizer.tokenize(content))
            else:
                print(f"Directory {batch_dir} does not exist.")

        self.all_cad_code = all_cad_code
        self.all_tokenized_texts = all_tokenized_texts
        self.rolls_range = torch.arange(-180, 180, 18)
        self.elevations_range = torch.arange(-90, 90, 10)

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
            shape=(self.batch_size, self.num_renders, 2),
        )
        rolls = self.rolls_range[random_angles[:, :, 0]]
        elevations = self.elevations_range[random_angles[:, :, 1]]
        random_cad_indices = torch.randint(
            low=0,
            high=len(self.all_tokenized_texts),
            shape=(self.batch_size,),
        )
        renderings = []
        for i in range(self.batch_size):
            renderings.append(
                render_cadquery_code(
                    self.all_cad_code[random_cad_indices[i]],
                    rolls[i],
                    elevations[i],
                )
            )

        return {
            "tokenized_texts": self.all_tokenized_texts[random_cad_indices],
            "rolls": rolls,
            "elevations": elevations,
            "cad_code": self.all_cad_code[random_cad_indices],
            "renderings": renderings,
        }

    def __iter__(self):
        while True:
            yield self._produce_batch()


class AutoRegressiveTrainer:
    def __init__(self, dataset, model, tokenizer):
        pass

    def loss_fn(self, output, target):
        pass

    def train(self, epochs):
        pass
