import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Set the device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_tensor(input_array):
    if not isinstance(input_array, torch.Tensor):
        input_array = torch.tensor(input_array, dtype=torch.float).to(device)
    return input_array

class AWE(nn.Module):
    """
    Encodes sentences by averaging the embeddings of tokens.
    """
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings):
        super(AWE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(ensure_tensor(pretrained_embeddings))
        self.embedding.weight.requires_grad = False
        self.output_size = embedding_dim

    def forward(self, text):
        embeddings = self.embedding(text.to(device))
        mean_embeddings = torch.mean(embeddings, dim=1)
        return mean_embeddings

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, pretrained_embeddings):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.embedding.weight.data.copy_(ensure_tensor(pretrained_embeddings))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=False, batch_first=True)
        self.output_size = hidden_size

    def forward(self, text):
        text = text.to(device)
        lengths = (text != 1).sum(dim=1)
        embeddings = self.embedding(text)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden_states, _) = self.lstm(packed)
        return hidden_states[-1]

class BiLSTM(nn.Module):
    """
    Encodes sentences by concatenating the last hidden states of a bidirectional LSTM.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, pretrained_embeddings):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.embedding.weight.data.copy_(ensure_tensor(pretrained_embeddings))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.output_size = hidden_size * 2

    def forward(self, text):
        text = text.to(device)
        lengths = (text != 1).sum(dim=1)
        embeddings = self.embedding(text)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden_states, _) = self.lstm(packed)
        return torch.cat((hidden_states[0], hidden_states[1]), dim=1)

class BiLSTM_MaxPool(nn.Module):
    """
    Encodes sentences using max pooling over the outputs of a bidirectional LSTM.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, pretrained_embeddings):
        super(BiLSTM_MaxPool, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.embedding.weight.data.copy_(ensure_tensor(pretrained_embeddings))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.output_size = hidden_size * 2

    def forward(self, text):
        text = text.to(device)
        embeddings = self.embedding(text)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        unpacked, _ = pad_packed_sequence(output, batch_first=True)
        max_pooled = torch.max(unpacked, dim=1).values
        return max_pooled

class Classifier(nn.Module):
    def __init__(self, encoder, mlp_hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(4 * encoder.output_size, mlp_hidden_size),
            nn.Tanh(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.Tanh(),
            nn.Linear(mlp_hidden_size, num_classes)
        )

    def forward(self, premise, hypothesis):
        premise = premise.to(device)
        hypothesis = hypothesis.to(device)
        p_encoded = self.encoder(premise)
        h_encoded = self.encoder(hypothesis)
        combined_features = torch.cat(
            [p_encoded, h_encoded, torch.abs(p_encoded - h_encoded), p_encoded * h_encoded],
            dim=1
        )
        logits = self.classifier(combined_features.to(device))
        return logits
