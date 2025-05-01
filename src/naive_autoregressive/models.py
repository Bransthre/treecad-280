import torch
import torch.nn as nn
from torchvision import models

from vocabularies import vocabularies


class Tokenizer(nn.Module):
    def __init__(self, vocabularies):
        super().__init__()
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.vocabularies = vocabularies + [
            self.pad_token,
            self.sos_token,
            self.eos_token,
        ]
        self.token_to_index = {token: i for i, token in enumerate(vocabularies)}
        self.index_to_token = {i: token for i, token in enumerate(vocabularies)}
        self.vocab_size = len(vocabularies)

        self.pad_token_id = self.token_to_index[self.pad_token]
        self.sos_token_id = self.token_to_index[self.sos_token]
        self.eos_token_id = self.token_to_index[self.eos_token]

    def tokenize(self, text):
        tokens = text.split()  # TODO: We need to do a bit more than this?
        token_ids = [
            self.token_to_index.get(token, self.pad_token_id) for token in tokens
        ]
        return token_ids

    def detokenize(self, token_ids):
        tokens = [
            self.index_to_token.get(token_id, self.pad_token) for token_id in token_ids
        ]
        return " ".join(tokens)


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
