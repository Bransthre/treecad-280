"""
This file contains the code for training an autoregressive model using a dataset of code snippets.
It will receive many images and output a sequence of tokens as the cadquery code.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import models

from vocabularies import vocabularies
from datrie import Trie


class Tokenizer:
    def __init__(self, vocabularies):
        self._pad_token = self._token_to_index["<PAD>"]
        self._sos_token = self._token_to_index["<SOS>"]
        self._eos_token = self._token_to_index["<EOS>"]
        self._vocabulary = ["<PAD>", "<SOS>", "<EOS>"] + vocabularies
        self._token_to_index = {token: i for i, token in enumerate(self._vocabulary)}
        self._index_to_token = {i: token for i, token in enumerate(self._vocabulary)}
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
        return token_indices

    def detokenize(self, token_indices):
        tokens = [
            self._index_to_token.get(token_id, self._pad_token)
            for token_id in token_indices
        ]
        return "".join(tokens)


class CodeGenerator(nn.Module):
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
        self.image_feedforward = nn.Linear(768, 512)
        self.tokenizer = Tokenizer(vocabularies)
        self.token_embedding = nn.Embedding(len(self.tokenizer._vocabulary), 512)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=6,
        )
        self.output_dim = output_dim

    def forward(self, images, tokenized):
        """
        A few arbitrary design decisions:
        (1) We will add the image embeddings together.
        Assume input is B x n_renderings x 3 x 224 x 224

        Function returns the token embeddings and the decoder output.
        """
        image_embeddings = self.encoder(images)
        image_embeddings = self.image_feedforward(image_embeddings.sum(dim=1))
        image_embeddings = image_embeddings.unsqueeze(1)
        decoder_input = self.token_embedding(tokenized)
        decoder_output = self.decoder(decoder_input, image_embeddings)
        return decoder_output


class AutoRegressiveDataset(IterableDataset):
    pass
