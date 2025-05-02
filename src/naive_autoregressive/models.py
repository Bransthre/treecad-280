import torch
import torch.nn as nn
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


class VisualEncoder(nn.Module):
    """
    The module naively takes in a pair of images (target rendering, current rendering)
    and outputs some representation.
    """

    def __init__(self, output_dim):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)

    def forward(self, image):
        return self.resnet(image)


class TextEncoder(nn.Module):
    """
    The module naively takes in some arbitrary text and outputs some representation.
    That text may be an incomplete CAD program, or a complete one when lucky.
    """

    def __init__(self, vocab_size, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, output_dim)
        self.lstm = nn.LSTM(output_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        return lstm_out


class CodeGenerator(nn.Module):
    """
    The module takes in image representation and text representation and outputs a sequence of tokens.
    The sequence of tokens is a CAD program.
    It joins representations by adding them.
    """

    def __init__(self, vocab_size, output_dim):
        super().__init__()
        self.fc = nn.Linear(output_dim * 2, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.tokenizer = Tokenizer(vocabularies)
        self.visual_encoder = VisualEncoder(output_dim)
        self.text_encoder = TextEncoder(vocab_size, output_dim)
        self.positional_encoding = nn.Embedding(512, output_dim)

    def forward(self, image, text):
        image_rep = self.visual_encoder(image)
        text_rep = self.text_encoder(text)
        positions = torch.arange(text.size(1), device=text.device).unsqueeze(0)
        pos_enc = self.positional_encoding(positions)
        text_rep = text_rep + pos_enc

        combined_rep = image_rep + text_rep
        logits = self.fc(combined_rep)
        probs = self.softmax(logits)
        return probs


# TODO: Make all of the above be transformer models
