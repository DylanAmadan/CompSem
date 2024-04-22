from datasets import load_dataset
from tqdm import tqdm
import nltk
import numpy as np
import json
import pickle
import torch
nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Set the seed to make the runs reproducible
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocess(entry):
    # Tokenize and process the text
    entry['text'] = nltk.word_tokenize(entry['text'].lower())
    return entry

def filter_invalid(entry):
    # Ensure only valid entries are retained
    return entry['label'] != -1

def fetch_and_process_data():
    """
    Fetches SNLI dataset and processes it through tokenization and filtering.
    Returns the processed dataset.
    """
    dataset = load_dataset("snli")
    dataset = dataset.map(tokenize_and_lower)  # Apply preprocessing
    dataset = dataset.filter(discard_invalid)  # Filter out invalid entries
    return dataset

def tokenize_and_lower(entry):
    """
    Tokenizes and converts text to lowercase.
    Returns the modified entry with tokenized and lowered premise and hypothesis.
    """
    for field in ['premise', 'hypothesis']:
        tokens = nltk.tokenize.word_tokenize(entry[field])
        entry[field] = [word.lower() for word in tokens]

    return entry

def discard_invalid(entry):
    """
    Discards entries where the label is -1.
    Returns True if the entry is valid, False otherwise.
    """
    return entry['label'] != -1

def debug_data_subset():
    """
    Loads a subset of the SNLI dataset for debugging purposes.
    Returns the subset containing only test data duplicated across train, validation, and test.
    """
    debug_dataset = load_dataset("snli")
    debug_dataset['test'] = debug_dataset['test'].map(tokenize_and_lower)
    debug_dataset['train'] = debug_dataset['test']
    debug_dataset['validation'] = debug_dataset['test']

    return debug_dataset

class WordIndex:
    """
    Manages the mapping of words to indices for dataset vocabularies.
    """
    def __init__(self, dataset):
        self.dataset_splits = {k: dataset[k] for k in ['train', 'validation', 'test']}
        self.index = {"<unk>": 0, "<pad>": 1}
        self.build_index()

    def build_index(self):
        idx = 2
        for data in self.dataset_splits.values():
            for entry in data:
                for token in entry['premise'] + entry['hypothesis']:
                    if token not in self.index:
                        self.index[token] = idx
                        idx += 1
        self.reverse_index = {v: k for k, v in self.index.items()}

    def get(self, word, default=0):
        return self.index.get(word, default)

def stream_glove_vectors(filepath):
    """
    Yields glove vectors line by line from the file.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            yield line.strip().replace("\\", "")

def create_embedding_matrix(filepath):
    """
    Constructs the embedding matrix using the glove vectors for known vocabulary words.
    """
    with open('data/data.json', 'r') as file:
        vocab_indices = json.load(file)

    matrix_size = len(vocab_indices)
    print(f"Vocabulary size: {matrix_size}")  
    embeddings = np.zeros((matrix_size, 300))
    embeddings[:2] = np.random.uniform(-1, 1, (2, 300))

    print("Starting embedding matrix creation.")
    for line in tqdm(stream_glove_vectors(filepath)):
        word, vector = line.split(" ", 1)
        if word in vocab_indices:
            embeddings[vocab_indices[word]] = np.array(vector.split(), dtype=np.float32)

    # Replace zero rows with the unknown token's vector
    zero_row_indices = np.all(embeddings == 0, axis=1)
    embeddings[zero_row_indices] = embeddings[0]

    with open("data/embedding_matrix.pickle", 'wb') as file:
        pickle.dump(embeddings, file)

    print("embedding matrix created.")
    return embeddings

def pad_sequence(tokens, length, pad_value=1):
    """
    Pads a sequence of tokens to a specified length with a given pad value.
    """
    return tokens + [pad_value] * (length - len(tokens))

def prepare_minibatch(batch, vocab):
    """
    Converts a batch of examples to tensors of word indices.
    """
    batch_size = len(batch)
    max_len = max(len(x['premise']) for x in batch) + max(len(x['hypothesis']) for x in batch)

    padded_premises = [pad_sequence([vocab.get(t, 0) for t in ex['premise']], max_len) for ex in batch]
    padded_hypotheses = [pad_sequence([vocab.get(t, 0) for t in ex['hypothesis']], max_len) for ex in batch]

    x_premise = torch.LongTensor(padded_premises).to(device)
    x_hypothesis = torch.LongTensor(padded_hypotheses).to(device)
    y_labels = torch.LongTensor([ex['label'] for ex in batch]).to(device)

    return x_premise, x_hypothesis, y_labels

def prep_example(example, vocab):
    """
    Map tokens to their IDs for a single example
    """
    
    # vocab returns 0 if the word is not there (i2w[0] = <unk>)
    x = [vocab.get(t, 0) for t in example]
    
    x = torch.LongTensor([x])
    x = x.to(device)

    return x

def load_embeddings(file_path):
    """
    Load the embedding matrix from a pickle file.
    
    Args:
    file_path (str): The path to the pickle file containing the embeddings.
    
    Returns:
    numpy.ndarray: The loaded embedding matrix.
    """
    with open(file_path, 'rb') as file:
        embeddings = pickle.load(file)
    return embeddings


if __name__ == "__main__":
    create_embedding_matrix("data/glove.840B.300d.txt")
