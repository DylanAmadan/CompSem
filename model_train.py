
import json

import pickle
import torch.optim as optim
import os

import argparse
import torch.nn as nn

import torch


from torch.utils.tensorboard import SummaryWriter
from data_loader import fetch_and_process_data, debug_data_subset, WordIndex, pad_sequence, load_embeddings, prepare_minibatch
from torch.utils.data import DataLoader
from alt_arch import Classifier, AWE, LSTM, BiLSTM, BiLSTM_MaxPool

from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def execute_training(model, session_name, train_dl, valid_dl, epochs, lr, device, save_dir):
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
    log_writer = SummaryWriter(f'logs/{session_name}')

    top_validation_accuracy = 0.0
    lr_min_threshold = 0.00001

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        train_losses, valid_losses = [], []
        total_correct_preds = 0

        for x1, x2, targets in train_dl:
            opt.zero_grad()
            predictions = model(x1, x2)
            loss = loss_fn(predictions, targets)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        scheduler.step()
        model.eval()
        with torch.no_grad():
            for x1, x2, targets in valid_dl:
                predictions = model(x1, x2)
                loss = loss_fn(predictions, targets)
                valid_losses.append(loss.item())
                total_correct_preds += (predictions.argmax(1) == targets).sum().item()

        valid_accuracy = total_correct_preds / len(valid_dl.dataset)
        log_metrics(epoch, train_losses, valid_losses, valid_accuracy, log_writer)

        if valid_accuracy > top_validation_accuracy:
            top_validation_accuracy = valid_accuracy
            # Save the model checkpoint
            checkpoint_path = os.path.join(save_dir, f'{session_name}_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model checkpoint saved to {checkpoint_path}')

        if lr < lr_min_threshold:
            print("Learning rate below threshold, stopping training.")
            break


def log_metrics(epoch, train_losses, valid_losses, valid_accuracy, writer):
    train_loss_avg = sum(train_losses) / len(train_losses)
    valid_loss_avg = sum(valid_losses) / len(valid_losses)
    writer.add_scalar('Loss/Train', train_loss_avg, epoch)
    writer.add_scalar('Loss/Valid', valid_loss_avg, epoch)
    writer.add_scalar('Accuracy/Valid', valid_accuracy, epoch)
    print(f"Epoch {epoch}: Train Loss = {train_loss_avg:.4f}, Valid Loss = {valid_loss_avg:.4f}, Validation Accuracy = {valid_accuracy:.4f}")

def adjust_learning_rate(optimizer, current_lr, reduction_factor):
    new_lr = current_lr / reduction_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def setup_model_and_data(args, device):
    # Fetch and process the dataset
    dataset = fetch_and_process_data()
    vocab = WordIndex(dataset)
    vocab.build_index()

    # Load the embeddings matrix
    embedding_matrix = load_embeddings('data/embedding_matrix.pickle')

    # Encoder types dictionary without device parameter in lambda
    encoder_types = {
        'AWE': lambda vocab_size, embedding_dim, pretrained_embeddings: AWE(vocab_size, embedding_dim, pretrained_embeddings),
        'LSTM': lambda vocab_size, embedding_dim, hidden_size, pretrained_embeddings: LSTM(vocab_size, embedding_dim, hidden_size, pretrained_embeddings),
        'BiLSTM': lambda vocab_size, embedding_dim, hidden_size, pretrained_embeddings: BiLSTM(vocab_size, embedding_dim, hidden_size, pretrained_embeddings),
        'BiLSTM_MaxPool': lambda vocab_size, embedding_dim, hidden_size, pretrained_embeddings: BiLSTM_MaxPool(vocab_size, embedding_dim, hidden_size, pretrained_embeddings)
    }

    if args.encoder in ['LSTM', 'BiLSTM', 'BiLSTM_MaxPool']:
        encoder = encoder_types[args.encoder](len(embedding_matrix), 300, args.hidden_size, embedding_matrix)
    else:
        encoder = encoder_types[args.encoder](len(embedding_matrix), 300, embedding_matrix)

    model = Classifier(encoder, 512, 3)
    model.to(device)

    # Create data loaders for training and validation sets
    train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, collate_fn=lambda batch: prepare_minibatch(batch, vocab), shuffle=True)
    valid_loader = DataLoader(dataset["validation"], batch_size=args.batch_size, collate_fn=lambda batch: prepare_minibatch(batch, vocab), shuffle=True)

    return model, train_loader, valid_loader


def main():
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the directory for saving model checkpoints
    model_save_dir = os.path.expanduser('~/Documents/Bigold/model_checkpoint/')
    os.makedirs(model_save_dir, exist_ok=True)  # Ensure the directory exists

    model, train_dl, valid_dl = setup_model_and_data(args, device)
    execute_training(model, args.session_name, train_dl, valid_dl, args.epochs, args.lr, device, model_save_dir)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network model on textual data.")
    parser.add_argument('--lr', default=0.1, type=float, help="Initial learning rate")
    parser.add_argument('--epochs', default=10, type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', default=64, type=int, help='Mini-batch size')
    parser.add_argument('--encoder', default='AWE', choices=['AWE', 'LSTM', 'BiLSTM', 'BiLSTM_MaxPool'], help="Type of encoder to use")
    parser.add_argument('--session_name', default='training_session', type=str, help="Name for the training logs and outputs")
    parser.add_argument('--hidden_size', default=256, type=int, help='Size of the LSTM hidden layer')
    return parser.parse_args()

if __name__ == "__main__":
    main()
