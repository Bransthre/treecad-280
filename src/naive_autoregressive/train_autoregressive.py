"""
This file contains the code for training an autoregressive model using a dataset of code snippets.
It will receive many images and output a sequence of tokens as the cadquery code.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision_vit import VisionTransformer, Encoder


import cadquery as cq
from cadquery import *
import matplotlib.pyplot as plt

from no_interaction_vis import no_interact_show
from vocabularies import vocabularies
from dataset import CADImageDataset, Tokenizer, collate_fn

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


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d):
        super().__init__()
        # Generate frequencies
        num_repeats = d // 32
        inv_freq = 1.0 / (2 ** (torch.arange(0, 32, 2).float() / 32))
        inv_freq = inv_freq.repeat_interleave(num_repeats)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len_dim):
        seq_len = x.size(seq_len_dim)
        pos = torch.arange(seq_len, dtype=torch.float, device=x.device)
        sinusoid_inp = torch.einsum("i , j -> ij", pos, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)

        # Expand for batch and sequence length
        emb = emb.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, hidden_dim)

        return (x * emb) + (x.flip(dims=[-1]) * emb)


class RollElevationEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, roll_idxs, elevation_idxs):
        high_dims_mults = torch.arange(
            0, self.d_model // 2, dtype=torch.float, device=roll_idxs.device
        )
        roll_idxs_expansion = roll_idxs[..., None] * high_dims_mults.repeat(
            (roll_idxs.shape[0], roll_idxs.shape[1], 1)
        )
        elevation_idxs_expansion = elevation_idxs[..., None] * high_dims_mults.repeat(
            (elevation_idxs.shape[0], elevation_idxs.shape[1], 1)
        )
        roll_idxs_expansion = torch.sin(roll_idxs_expansion)
        elevation_idxs_expansion = torch.cos(elevation_idxs_expansion)
        roll_elevation_embeddings = torch.cat(
            (roll_idxs_expansion, elevation_idxs_expansion), dim=-1
        )  # (B, d_model)
        return roll_elevation_embeddings


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
            num_layers=8,
            num_heads=8,
            hidden_dim=512,
            mlp_dim=2048,
            in_channels=3,
        )  # I don't know if this is the right parameters but here we go.
        self.rope_embedding = RotaryPositionalEmbeddings(d=512)
        self.tokenizer = Tokenizer(vocabularies)
        self.token_embedding = nn.Embedding(len(self.tokenizer._vocabulary), 768)
        self.roll_elevation_embedding = RollElevationEmbedding(256)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=768, nhead=8, batch_first=True
            ),  # batch_first is True automatically
            num_layers=6,
        )
        self.out_logits = nn.Linear(768, output_dim)

    def forward(self, renderings, tokenized, roll_idxs, elevation_idxs, tgt_mask):
        """
        A few arbitrary design decisions, which are all heavily dependent on
        (tree-diffusion/td/learning/gpt.py).

        (1) We will concatenate (meaning 3k channels) the k provided renderings.
        They should probably accompany some angular positional embedding here.
        Assume input `renderings` is B x k x C x 128 x 128
        """
        renderings = renderings.permute(0, 1, 4, 2, 3) * 2 - 1
        image_embeddings = self.encoder(
            renderings.reshape(-1, 3, 128, 128)
        )  # B x k x 768 x 128
        image_embeddings = self.rope_embedding(image_embeddings[None], 2)
        decoder_input = self.token_embedding(tokenized)
        roll_elevation_embeddings = self.roll_elevation_embedding(
            roll_idxs, elevation_idxs
        )
        image_embeddings = image_embeddings.reshape(-1, image_embeddings.shape[-2], 512)
        roll_elevation_embeddings = (
            roll_elevation_embeddings[:, :, None, :]
            .repeat((1, 1, image_embeddings.shape[-2], 1))
            .reshape(-1, image_embeddings.shape[-2], 256)
        )
        visual_embeddings = torch.cat(
            (image_embeddings, roll_elevation_embeddings), dim=-1
        ).permute(
            1, 0, 2
        )  # B x T x (768 + 256)
        visual_embeddings = visual_embeddings.reshape(
            renderings.shape[0], renderings.shape[1], -1, 768
        ).mean(dim=1)
        subsequent_mask = torch.triu(
            torch.full(
                (decoder_input.shape[1], decoder_input.shape[1]),
                float("-inf"),
                device=decoder_input.device,
            ),
            diagonal=1,
        )
        visual_embeddings = visual_embeddings.repeat(
            1, decoder_input.shape[1] // visual_embeddings.shape[1] + 1, 1
        )
        visual_embeddings = visual_embeddings[:, : decoder_input.shape[1], :]

        # print(f"There are {decoder_input.isnan().sum()} NaNs in the decoder input")
        # print(
        #     f"There are {visual_embeddings.isnan().sum()} NaNs in the visual embeddings"
        # )
        # print(f"There are {subsequent_mask.isnan().sum()} NaNs in the subsequent mask")
        # print(f"There are {tgt_mask.isnan().sum()} NaNs in the tgt mask")

        # print(f"The shape of decoder_input is {decoder_input.shape}")
        # print(f"The shape of visual_embeddings is {visual_embeddings.shape}")
        # print(f"The shape of subsequent_mask is {subsequent_mask.shape}")
        # print(f"The shape of tgt_mask is {tgt_mask.shape}")

        decoder_output = self.decoder(
            decoder_input,
            visual_embeddings,
            # tgt_mask=subsequent_mask,
            tgt_key_padding_mask=tgt_mask,
        )
        decoder_output_logits = self.out_logits(decoder_output)
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


def model_inference(
    model: BaselineCADGenerator, renders, rolls, elevations, tgt_masks, tokenizer
):
    beam_size = 5
    batch_size = renders.shape[0]
    seq_len = 512

    # Initialize best sequence contexts with the start token for beam search
    best_sequence_contexts = torch.full(
        (beam_size, 1), tokenizer._sos_token, dtype=torch.long
    ).cuda()  # (B, beam_size, 1)

    # Initialize beam scores (log probabilities of sequences)
    beam_scores = torch.zeros(beam_size).cuda()  # (B, beam_size)
    renders_repeat = renders.repeat(
        (beam_size, 1, 1, 1, 1)
    )  # (B * beam_size, k, 3, 128, 128)
    rolls_repeat = rolls.repeat(beam_size, 1)  # (B * beam_size,)
    elevations_repeat = elevations.repeat(beam_size, 1)  # (B * beam_size,)

    print("renders.shape", renders_repeat.shape)
    print(
        "tokenized_input",
        best_sequence_contexts.reshape(-1, best_sequence_contexts.shape[-1]).shape,
    )
    print("rolls.shape", rolls_repeat.shape)
    print("elevations.shape", elevations_repeat.shape)

    for i in range(seq_len):
        print("best_sequence_contexts.shape", best_sequence_contexts.shape)
        padding_mask = torch.zeros((beam_size, best_sequence_contexts.shape[-1])).cuda()

        # Forward pass with current best sequences
        next_token_logits = model(
            renders_repeat,
            best_sequence_contexts,
            rolls_repeat,
            elevations_repeat,
            padding_mask,
        )  # Reshape to (B, beam_size, vocab_size)

        # Apply beam search: get the top `beam_size` token indices
        print("L236 next_token_logits.shape", next_token_logits.shape)
        next_token_logits = next_token_logits[
            :, :, :-1
        ]  # Remove unwanted dimension (vocab size) if needed
        print("L240 next_token_logits.shape", next_token_logits.shape)
        log_probs = torch.log_softmax(next_token_logits, dim=-1)  # Log probabilities
        print("log_probs.shape", log_probs.shape)

        # Compute total log probability scores for beam search
        beam_scores = (
            beam_scores.unsqueeze(-1) + log_probs
        )  # (B, beam_size, vocab_size)

        # Select top `beam_size` from each batch
        print("beam_scores:", beam_scores.shape)
        top_scores, top_indices = torch.topk(beam_scores.view(-1), beam_size, dim=-1)

        # Extract the corresponding batch and beam indices
        beam_idx = top_indices // tokenizer._vocabulary_size  # Get the batch index
        token_idx = top_indices % tokenizer._vocabulary_size  # Get the token index

        # Update the sequences
        best_sequence_contexts = torch.cat(
            [
                best_sequence_contexts[beam_idx],
                token_idx.unsqueeze(-1),
            ],
            dim=-1,
        )

        # Update the scores
        beam_scores = top_scores

        # Print shapes for debugging
        print("best_sequence_contexts.shape", best_sequence_contexts.shape)
        print("beam_scores.shape", beam_scores.shape)

    # Convert to list of decoded tokens
    best_sequence_contexts = best_sequence_contexts.squeeze().cpu().numpy()  # (B, T)
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
    dataset = CADImageDataset()
    model = BaselineCADGenerator(output_dim=config["vocab_size"]).cuda()
    if config["ckpt"] is not None:
        model.load_state_dict(
            torch.load("model_weights.pth", map_location="cuda:0", weights_only=True)
        )
    model = nn.DataParallel(model)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.Adam(lr=config["lr"], params=model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    step = config["restore_step"] if config["restore_step"] is not None else 0

    run = wandb.init(
        **{"entity": "brandonh", "project": "280-project", "group": "debugging"}
    )
    for i in tqdm(range(1, config["total_steps"] + 1), desc="Training epochs..."):
        within_step_batch_bar = tqdm(
            range(len(dataloader)),
            desc=f"Training batches...",
            leave=False,
            total=len(dataloader),
        )
        for batch in dataloader:
            tokenized_inputs = batch["decoder_inputs"].cuda()  # (B, 512)
            targets = batch["targets"].cuda()  # (B, 512)
            tgt_masks = batch["attention_masks"].cuda()  # (B, 512)
            rolls = batch["rolls"].squeeze().cuda()  # (B,)
            elevations = batch["elevations"].squeeze().cuda()  # (B,)
            renderings = batch["renderings"].cuda()  # tensor (B x k x 3 x 128 x 128)
            optimizer.zero_grad()
            outputs = model(renderings, tokenized_inputs, rolls, elevations, tgt_masks)

            loss = criterion(
                outputs.reshape(-1, config["vocab_size"]),
                targets.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss = loss.item()
            wandb.log({"loss/train_loss": loss}, step)

            if step % config["eval_steps"] == 0:
                model.eval()
                evaluate(model, config, step, dataset.tokenizer)
                model.train()

            if step % config["save_steps"] == 0:
                torch.save(
                    model.module.state_dict(), f"model_weights_{run.name}_{step}.pth"
                )
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


def evaluate(
    model: BaselineCADGenerator, config: dict, step: int, tokenizer: Tokenizer
):
    dataset = CADImageDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    criterion = nn.CrossEntropyLoss()

    sample_render = None

    losses = []
    for batch in dataloader:
        tokenized_inputs = batch["decoder_inputs"].cuda()  # (B, 512)
        targets = batch["targets"].cuda()  # (B, 512)
        tgt_masks = batch["attention_masks"].cuda()  # (B, 512)
        rolls = batch["rolls"].squeeze().cuda()  # (B,)
        elevations = batch["elevations"].squeeze().cuda()  # (B,)
        renderings = batch["renderings"].cuda()  # tensor (B x k x 3 x 128 x 128)

        outputs = model(renderings, tokenized_inputs, rolls, elevations, tgt_masks)

        # if sample_render is None:
        #     with torch.no_grad():
        #         cadquery_code = model_inference(
        #             model,
        #             renderings[0:1],
        #             rolls[0:1],
        #             elevations[0:1],
        #             tgt_masks[0:1],
        #             tokenizer,
        #         )

        #     images = render_cadquery_code(
        #         cadquery_code[0],
        #         rolls[0].unsqueeze(0),
        #         elevations[0].unsqueeze(0),
        #     )

        #     log_renders(images)

        loss = criterion(
            outputs.reshape(-1, config["vocab_size"]),
            targets.reshape(-1),
        )

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
