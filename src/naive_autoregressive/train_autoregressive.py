"""
This file contains the code for training an autoregressive model using a dataset of code snippets.
It will receive many images and output a sequence of tokens as the cadquery code.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision_vit import VisionTransformer


import cadquery as cq
from cadquery import *
import matplotlib.pyplot as plt

from model_utils import RotaryPositionalEmbeddings
from no_interaction_vis import no_interact_show
from vocabularies import vocabularies
from dataset import AutoRegressiveDataset, Tokenizer

import os
import yaml
import wandb
from tqdm import tqdm
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "yaml_file_path", "/home/brandonh/src/config/default_config.yaml", "yaml file path"
)
flags.DEFINE_float("flt", 0.0, "")
flags.DEFINE_integer("batch_size", 512, "batch size")


class RollElevationEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.roll_embedding = nn.Embedding(20, d_model)
        self.elevation_embedding = nn.Embedding(20, d_model)

    def forward(self, roll_idxs, elevation_idxs):
        # Alternative iis to just use sinusoidal embeddings.
        roll_ohe = torch.nn.functional.one_hot(roll_idxs, num_classes=20)
        elevation_ohe = torch.nn.functional.one_hot(elevation_idxs, num_classes=20)
        roll_embedding = self.roll_embedding(roll_ohe)
        elevation_embedding = self.elevation_embedding(elevation_ohe)
        return torch.cat((roll_embedding, elevation_embedding), dim=-1)


class BaselineCADGenerator(nn.Module):
    """
    The module naively takes in a pair of images (target rendering, current rendering)
    and outputs a sequence of tokens resembling the CAD file code.
    """

    def __init__(self, output_dim):
        super().__init__()
        self.encoder = VisionTransformer(
            image_size=128,
            patch_size=16,
            num_classes=output_dim,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            in_channels=3 * 8,
        )  # I don't know if this is the right parameters but here we go.
        self.tokenizer = Tokenizer(vocabularies)
        self.token_embedding = nn.Embedding(len(self.tokenizer._vocabulary), 768)
        self.roll_elevation_embedding = RollElevationEmbedding(256)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=1024, nhead=8, batch_first=True),
            num_layers=6,
        )
        self.output_dim = output_dim
        self.rope_embedding = RotaryPositionalEmbeddings(d=768)

    def forward(self, renderings, tokenized, roll_idxs, elevation_idxs):
        """
        A few arbitrary design decisions, which are all heavily dependent on
        (tree-diffusion/td/learning/gpt.py).

        (1) We will concatenate (meaning 3k channels) the k provided renderings.
        They should probably accompany some angular positional embedding here.
        Assume input `renderings` is n_workers x B x k x C x 128 x 128
        """
        renderings = renderings.permute(0, 1, 2, 5, 3, 4)
        concat_img_repr = torch.reshape(
            renderings,
            (
                renderings.shape[0] * renderings.shape[1],
                -1,
                renderings.shape[-2],
                renderings.shape[-1],
            ),
        )
        concat_img_repr = concat_img_repr * 2 - 1
        image_embeddings = self.encoder(concat_img_repr)
        image_embeddings = self.rope_embedding(image_embeddings)
        decoder_input = self.token_embedding(tokenized)
        roll_elevation_embeddings = self.roll_elevation_embedding(
            roll_idxs, elevation_idxs
        )
        visual_embeddings = torch.cat(
            (image_embeddings, roll_elevation_embeddings), dim=-1
        )  # B x (768 + 256)
        decoder_output_logits = self.decoder(decoder_input, visual_embeddings)
        return decoder_output_logits


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


def model_inference(model: BaselineCADGenerator, renders, rolls, elevations):
    assert len(renders.shape) == 2
    tokenizer = model.tokenizer
    beam_size = 5
    best_sequence_contexts = torch.full(
        (renders.shape[0], beam_size, 1), tokenizer._sos_token, dtype=torch.long
    ).cuda()  # (B, beam_size, T)
    seq_len = 512
    for i in range(seq_len):
        next_token_logits = model(
            renders,
            best_sequence_contexts.reshape(-1, best_sequence_contexts.shape[-1]),
            rolls,
            elevations,
        ).reshape(
            renders.shape[0], -1
        )  # (B, 5*53)
        best_sequence_indices = torch.topk(next_token_logits, beam_size, dim=-1).indices
        best_next_token_idxs = torch.topk(next_token_logits, beam_size, dim=-1).indices
        best_contexts = best_sequence_indices // tokenizer._vocabulary_size
        best_sequence_contexts = best_sequence_contexts[
            torch.arange(best_sequence_contexts.shape[0]).unsqueeze(1),
            best_contexts,
            :,
        ]
        best_sequence_contexts = torch.cat(
            (
                best_sequence_contexts,
                best_next_token_idxs.unsqueeze(-1),
            ),
            dim=-1,
        )
        # best_sequence_contexts should be (B, beam_size, T)

    best_sequence_contexts = best_sequence_contexts.squeeze().cpu().numpy()
    decoded_texts = []
    for i in range(best_sequence_contexts.shape[0]):
        decoded_texts.append(
            [
                tokenizer._index_to_token.get(token_id, tokenizer._pad_token)
                for token_id in best_sequence_contexts[i]
            ]
        )
    decoded_texts = ["".join(decoded_text) for decoded_text in decoded_texts]
    return decoded_texts


def train(config):

    # gpus = list(range(torch.cuda.device_count()))
    # print(gpus)
    dataset = AutoRegressiveDataset(
        batch_size=config["batch_size"], num_renders=config["num_renders"]
    )
    model = BaselineCADGenerator(output_dim=config["vocab_size"]).cuda()
    if config["ckpt"] is not None:
        model.load_state_dict(
            torch.load("model_weights.pth", map_location="cuda:0", weights_only=True)
        )
    model = nn.DataParallel(model)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        drop_last=True,
    )

    optimizer = torch.optim.Adam(lr=config["lr"], params=model.parameters())
    criterion = nn.CrossEntropyLoss()

    step = config["restore_step"] if config["restore_step"] is not None else 0

    wandb.init(**{"entity": "brandonh", "project": "280-project", "group": "debugging"})
    for i in tqdm(range(1, config["total_steps"] + 1), desc="Training steps..."):
        within_step_batch_bar = tqdm(
            range(len(dataloader)),
            desc=f"Training batches...",
            leave=False,
            total=len(dataloader),
        )
        for batch in dataloader:
            tokenized_texts = batch["tokenized_texts"].cuda()  # (B, 512, d)
            rolls = batch["rolls"].squeeze().cuda()  # (B,)
            elevations = batch["elevations"].squeeze().cuda()  # (B,)
            renderings = batch["renderings"].cuda()  # tensor (B x k x 3 x 128 x 128)
            optimizer.zero_grad()
            outputs = model(renderings, tokenized_texts[:, :-1, :], rolls, elevations)
            loss = criterion(outputs, tokenized_texts[:, 1:, :])
            loss.backward()
            optimizer.step()

            loss = loss.item()
            wandb.log({"loss/train_loss": loss}, step)

            if step % config["eval_steps"] == 0:
                model.eval()
                evaluate(model, config, step)
                model.train()

            if step % config["save_steps"] == 0:
                torch.save(model.module.state_dict(), "model_weights.pth")
            within_step_batch_bar.update(1)

            step += 1
        within_step_batch_bar.close()


def log_renders(images):
    images_cl = [
        np.transpose(img, (1, 2, 0)) for img in images[:8]
    ]  # Ensure (128, 128, 3)

    # Create a 4x2 subplot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.imshow(np.clip(images_cl[i], 0, 1))  # Clip values to [0, 1] for safety
        ax.axis("off")

    plt.tight_layout()

    # Log the figure to wandb
    wandb.log({"image_grid": wandb.Image(fig)})

    # Close the plot to avoid memory leaks in loops
    plt.close(fig)


def evaluate(model: BaselineCADGenerator, config: dict, step: int):
    dataset = AutoRegressiveDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        # num_workers=8,
        drop_last=True,
    )
    criterion = nn.CrossEntropyLoss()

    sample_render = None

    losses = []
    for batch in dataloader:
        tokenized_texts = batch["tokenized_texts"].cuda()  # (B, 512, d)
        rolls = batch["rolls"].squeeze().cuda()  # (B,)
        elevations = batch["elevations"].squeeze().cuda()  # (B,)
        renderings = batch["renderings"].cuda()  # tensor (B x k x 3 x 128 x 128)
        outputs = model(renderings, tokenized_texts[:, :-1, :], rolls, elevations)

        if sample_render is None:
            with torch.no_grad():
                cadquery_code = model_inference(
                    model,
                    renderings[0].unsqueeze(0),
                    rolls[0].unsqueeze(0),
                    elevations[0].unsqueeze(0),
                )

            images = render_cadquery_code(
                cadquery_code[0],
                rolls[0].unsqueeze(0),
                elevations[0].unsqueeze(0),
            )

            log_renders(images)

        loss = criterion(outputs, tokenized_texts[:, 1:, :])

        loss = loss.item()
        losses.append(loss)
    loss = sum(losses) / len(losses)
    wandb.log({"loss/eval_loss": loss}, step)
    return


if __name__ == "__main__":
    config = yaml.safe_load(
        open("/home/brandonh/treeCAD/src/config/default_config.yaml", "r")
    )
    train(config)
