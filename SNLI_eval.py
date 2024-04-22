import torch
from torch import nn
import argparse
import json
from data_loader import fetch_and_process_data, prepare_minibatch, load_embeddings
from new_arch import Classifier, AWE, LSTM, BiLSTM, BiLSTM_MaxPool
from torch.utils.data import DataLoader
from functools import partial
import numpy as np

def compute_model_metrics(network, loader):
    loss_evaluator = nn.CrossEntropyLoss()
    network.eval()
    cumulative_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for premises, hypotheses, labels in loader:
            predictions = network(premises, hypotheses)
            loss = loss_evaluator(predictions, labels)
            cumulative_loss += loss.item()
            correct_predictions += (predictions.argmax(dim=1) == labels).type(torch.float).sum().item()
    average_loss = cumulative_loss / len(loader)
    total_accuracy = correct_predictions / len(loader.dataset)
    return average_loss, total_accuracy

def main():
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument('--model_path', required=True, help="File path to the trained model.")
    parser.add_argument('--batch_size', default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument('--encoder_type', required=True, choices=['AWE', 'LSTM', 'BiLSTM', 'BiLSTM_MaxPool'], help="Type of encoder: AWE, LSTM, BiLSTM, or BiLSTM_MaxPool.")

    args = parser.parse_args()
    computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    test_data = fetch_and_process_data()

    # Load vocabulary and embeddings
    vocab_w2i = json.load(open('data/data.json', 'r'))
    embeddings_tensor = load_embeddings('data/embedding_matrix.pickle')

    # Convert embeddings tensor to numpy if required by the architecture file
    embeddings_array = embeddings_tensor.cpu().numpy() if isinstance(embeddings_tensor, torch.Tensor) else embeddings_tensor

    # Initialize encoder based on the encoder_type argument
    encoders = {
        'AWE': lambda: AWE(len(embeddings_array), 300, embeddings_array, computation_device),
        'LSTM': lambda: LSTM(len(embeddings_array), 300, 2048, embeddings_array, computation_device),
        'BiLSTM': lambda: BiLSTM(len(embeddings_array), 300, 2048, embeddings_array, computation_device),
        'BiLSTM_MaxPool': lambda: BiLSTM_MaxPool(len(embeddings_array), 300, 2048, embeddings_array, computation_device)
    }
    encoder = encoders[args.encoder_type]()
    model = Classifier(encoder, mlp_hidden_size=512, num_classes=3, device=computation_device)

    # Load the state dictionary into the model
    state_dict = torch.load(args.model_path, map_location=computation_device)
    model.load_state_dict(state_dict)
    model.to(computation_device)

    # Prepare DataLoader
    collate_fn = partial(prepare_minibatch, vocab=vocab_w2i)
    test_loader = DataLoader(test_data["test"], batch_size=args.batch_size, collate_fn=collate_fn)

    # Evaluate the model
    test_loss, test_accuracy = compute_model_metrics(model, test_loader)
    print(f"Test Metrics: Loss={test_loss:.4f}, Accuracy={test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
