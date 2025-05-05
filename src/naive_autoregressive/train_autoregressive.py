"""
This file contains the code for training an autoregressive model using a dataset of code snippets.
It will receive many images and output a sequence of tokens as the cadquery code.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import models

from .model_utils import RotaryPositionalEmbeddings

import cadquery as cq
from cadquery import *
from no_interaction_vis import no_interact_show
import matplotlib.pyplot as plt

from .vocabularies import vocabularies
from datrie import Trie

import os
import yaml
import wandb
from tqdm import tqdm
from absl import flags

# TODO: Set up wandb logging

FLAGS = flags.FLAGS
flags.DEFINE_string("yaml_file_path", "", "yaml file path")
flags.DEFINE_float("flt", 0.0, "")
flags.DEFINE_integer("batch_size", 512, "batch size")


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
        # self.rope_embedding = RotaryPositionalEmbeddings(dim=768)
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
        concat_img_repr = concat_img_repr * 2 - 1
        # concat_img_repr = self.rope_embedding(concat_img_repr)
        image_embeddings = self.encoder(concat_img_repr)  # Normalize to [-1, 1]
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

    cq_namespace = {"r": None}
    exec("import cadquery as cq;" + cadquery_code, cq_namespace)
    for img_idx, (roll, elevation) in enumerate(zip(rolls, elevations)):
        no_interact_show(
            cq_namespace["r"],
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
            dataset_dir = "/home/brandonh/cad-recode-v1.5/train/"
            batch_ids = [f"0{i}" for i in range(10)] + list(range(10, 100))
            for batch_id in tqdm(batch_ids, desc="Processing batches", leave=False):
                batch_dir = os.path.join(dataset_dir, f"batch_{batch_id}")

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
        else:
            dataset_dir = "/home/brandonh/cad-recode-v1.5/val/"
            for _, file_name in tqdm(
                enumerate(os.listdir(dataset_dir)),
                desc=f"Processing files in {dataset_dir}",
            ):
                file_path = os.path.join(dataset_dir, file_name)
                if os.path.isfile(file_path):
                    with open(file_path, "r") as file:
                        contents = "".join(
                            [s.replace("\n", ";") for s in file.readlines()[1:]]
                        )
                        all_cad_code.append(contents)
                        all_tokenized_texts.append(self.tokenizer.tokenize(contents))

        self.all_cad_code = all_cad_code
        self.all_tokenized_texts = torch.Tensor(all_tokenized_texts)
        self.rolls_range = torch.arange(-180, 180, 18)
        self.elevations_range = torch.arange(-90, 90, 9)

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
        random_cad_indices = torch.randint(
            low=0,
            high=len(self.all_tokenized_texts),
            size=(self.batch_size,),
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
        renderings = torch.Tensor(np.array(renderings))  # (B x k x 3 x 224 x 224)

        return {
            "tokenized_texts": self.all_tokenized_texts[random_cad_indices],
            "rolls": rolls,
            "elevations": elevations,
            "renderings": renderings,
        }

    def __iter__(self):
        while True:
            yield self._produce_batch()


def train(config):
    wandb.init({"entity": "brandonh", "project": "280-project", "config": {}})

    gpus = list(range(torch.cuda.device_count()))

    dataset = AutoRegressiveDataset()
    model = BaselineCADGenerator(output_dim=config["vocab_size"]).cuda()
    if config["ckpt"] is not None:
        model.load_state_dict(
            torch.load("model_weights.pth", map_location="cuda:0", weights_only=True)
        )
    model = nn.DataParallel(model, gpus)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    step = config["restore_step"] if config["restore_step"] is not None else 0

    for i in tqdm(range(1, config["total_steps"] + 1)):
        for batch in dataloader:
            tokenized_texts = batch["tokenized_texts"].cuda()  # (B, 512, d)
            # rolls = batch["rolls"].squeeze().cuda()  # (B,)
            # elevations = batch["elevations"].squeeze().cuda()  # (B,)
            renderings = batch["renderings"].cuda()  # tensor (B x k x 3 x 224 x 224)
            optimizer.zero_grad()
            outputs = model(renderings, tokenized_texts[:, :-1, :])
            loss = criterion(outputs, tokenized_texts[:, 1:, :])
            loss.backward()
            optimizer.step()

            loss = loss.item()
            wandb.log({"loss": loss}, step)

            if step % config["eval_steps"] == 0:
                evaluate(model)

            if step % config["save_steps"] == 0:
                torch.save(model.module.state_dict(), "model_weights.pth")

            step += 1


def evaluate(model, config):
    dataset = AutoRegressiveDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=8,
        drop_last=True,
    )

    return


if __name__ == "__main__":
    train()
