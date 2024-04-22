import torch

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class AWE(nn.Module):
    """
    Encodes sentences by averaging the embeddings of tokens.
    """
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings, device):
        super(AWE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        if isinstance(pretrained_embeddings, np.ndarray):
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = False

        self.output_size = embedding_dim  # Assuming the output size is the embedding dimension

    def forward(self, text):
        embeddings = self.embedding(text)
        mean_embeddings = torch.mean(embeddings, dim=1)
        return mean_embeddings

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, pretrained_embeddings, device):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1).to(device)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=False, batch_first=True).to(device)
        self.output_size = hidden_size

    def forward(self, text):
        lengths = (text != 1).sum(dim=1)
        embeddings = self.embedding(text)
        packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden_states, _) = self.lstm(packed)
        return hidden_states[-1]


class BiLSTM(nn.Module):
    """
    Encodes sentences by concatenating the last hidden states of a bidirectional LSTM.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, pretrained_embeddings, device):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1).to(device)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True).to(device)
        self.output_size = hidden_size * 2

    def forward(self, text):
        lengths = (text != 1).sum(dim=1)
        embeddings = self.embedding(text)
        packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden_states, _) = self.lstm(packed)
        return torch.cat((hidden_states[0], hidden_states[1]), dim=1)

class BiLSTM_MaxPool(nn.Module):
    """
    Encodes sentences using max pooling over the outputs of a bidirectional LSTM.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, pretrained_embeddings, device):
        super(BiLSTM_MaxPool, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1).to(device)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True).to(device)
        self.output_size = hidden_size * 2

    def forward(self, text):
        lengths = (text != 1).sum(dim=1)
        embeddings = self.embedding(text)
        packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        unpacked, _ = pad_packed_sequence(output, batch_first=True)
        max_pooled = torch.max(unpacked, dim=1).values
        return max_pooled

class Classyfer(nn.Module):
    def __init__(self, encoder, mlp_hidden_size, num_classes, device):
        super(Classyfer, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(4 * encoder.output_size, mlp_hidden_size),  # Assuming encoder.output_size is 256
            nn.Tanh(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.Tanh(),
            nn.Linear(mlp_hidden_size, num_classes)
        ).to(device)

    def forward(self, premise, hypothesis):
        p_encoded = self.encoder(premise)
        h_encoded = self.encoder(hypothesis)
        # Assuming p_encoded and h_encoded are 256 dimensions each
        combined_features = torch.cat(
            [p_encoded, h_encoded, torch.abs(p_encoded - h_encoded), p_encoded * h_encoded], dim=1
        )  # This should create a tensor with 1024 dimensions
        print("Combined features shape:", combined_features.shape)
        logits = self.classifier(combined_features)
        return logits


"""

class Classyfer(nn.Module):
    def __init__(self, encoder, mlp_hidden_size, num_classes, device):
        super(Classyfer, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(4 * encoder.output_size, mlp_hidden_size),
            nn.Tanh(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.Tanh(),
            nn.Linear(mlp_hidden_size, num_classes)
        ).to(device)

    def forward(self, premise, hypothesis):
        p_encoded = self.encoder(premise)
        h_encoded = self.encoder(hypothesis)
        combined_features = torch.cat(
            (p_encoded, h_encoded, torch.abs(p_encoded - h_encoded), p_encoded * h_encoded),
            dim=1
        )
        logits = self.classifier(combined_features)
        return logits
    
"""