# CompSem -- ATCS Practical 1

The first practical of the Advanced Topics in Computational Semantics course delves into the realm of learning general-purpose sentence representations within the context of natural language inference (NLI). Our objective encompasses:

- Implementing four neural models to classify sentence pairs based on their relation.
- Training these models utilizing the Stanford Natural Language Inference (SNLI) corpus (Bowman et al., 2015).
- Evaluating the trained models using the SentEval framework (Conneau and Kiela, 2018).

NLI involves discerning entailment or contradiction relationships between premises and hypotheses, a fundamental aspect of understanding language. This assignment emphasizes pretraining a sentence encoder on NLI and subsequently evaluating its efficacy on diverse natural language tasks.

## Pre-trained Models
Download the pre-trained models from the following Google Drive link:
[Pre-trained Models Download](https://drive.google.com/drive/folders/1-sZ6OJudssRCEkijydCYbY5Zn2QY-Yub?usp=sharing)

## Environment and Dependencies

conda env create -f enironment2.yml
conda activate YourEnvironmentName


### Key Python Files:
- **data.py**: Manages the SNLI dataset loading, preprocessing (tokenizing and lowercasing), and embedding matrix creation. The matrix, which includes pre-trained Glove embeddings for the SNLI vocabulary, is stored in `data/embedding_matrix.pickle`. A JSON dictionary of the vocabulary is saved in `data/data.json`.
- **model.py**: Defines the neural network models, including various encoders and a classifier:
  - `AWE`: Averages all embeddings of a sentence to provide a sentence representation.
  - `LSTM`: Utilizes the last hidden state of a forward LSTM for sentence representation.
  - `BiLSTM`: Uses the concatenated last hidden states from a bidirectional LSTM.
  - `BiLSTM_MaxPool`: Applies max pooling across concatenated hidden states from a bidirectional LSTM.
  - `Classifyer`: Employs an encoder to generate a relation vector, which is then processed by a multi-layer perceptron.
- **train_model.py**: Script for model training, accepting various flags (e.g., learning rate, epochs, batch size, encoder type, model name).
- **SNLLI_eval.py**: Evaluates the trained model against the SNLI test dataset. Requires a checkpoint path flag.


## Training and Evaluation Process
1. **Preparation**:
   - Download and unzip the Glove embeddings from [Glove's Official Website](https://nlp.stanford.edu/projects/glove/). Place them in `Practical1/data`.
   - Execute `data.py` to generate and store the embedding matrix.
2. **Model Training**:
   - Select a model configuration and start the training using `model_train.py`, specifying the necessary flags.
3. **Model Evaluation**:
   - Evaluate the model on the SNLI dataset using `SNLI_eval.py`.

