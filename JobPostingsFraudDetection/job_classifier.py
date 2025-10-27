import os
import nltk
import string
import torch
import torch.nn as nn
import numpy as np
from bisect import bisect_right
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from job_classification import prepare_dataset
import math
import torch.optim as optim


DEVICE = torch.device('cuda')  # change this to 'cuda' if you want to use GPU
PAD_IDX = 0
CLS_TOKEN = '[CLS]'  # the special [CLS] token to be prepended to each sequence
SEP_TOKEN = '[SEP]'
SEED = 4065

tokeniser = nltk.tokenize.TreebankWordTokenizer()
stopwords = frozenset(nltk.corpus.stopwords.words("english"))
trans_table = str.maketrans(dict.fromkeys(string.punctuation))


def tokenise_text(str_):
    """Tokenize a string of text.

    Args:
        str_: The input string of text.

    Returns:
        list(str): A list of tokens.
    """
    # for simplicity, remove non-ASCII characters
    str_ = str_.encode(encoding='ascii', errors='ignore').decode()
    return [t for t in tokeniser.tokenize(str_.lower().translate(trans_table)) if t not in stopwords]


def build_vocab(Xt, min_freq=1):
    """Create a list of sentences, build the vocabulary and compute word frequencies from the given text data.

    Args:
        Xr (iterable(str)): A list of strings each representing a document.
        min_freq: The minimum frequency of a token that will be kept in the vocabulary.

    Returns:
        vocab (dict(str : int)): A dictionary mapping a word/token to its index.
    """
    print('Building vocabulary ...')
    counter = Counter()
    for xt in Xt:
        counter.update(xt)
    sorted_token_freq_pairs = counter.most_common()

    # find the first index where freq=min_freq-1 in sorted_token_freq_pairs using binary search/bisection
    # end = bisect_right(sorted_token_freq_pairs, -min_freq, key=lambda x: -x[1]) # Lead to TypeError

    # Find the index where freq < min_freq
    end = next((i for i, (_, freq) in enumerate(sorted_token_freq_pairs) if freq < min_freq), len(sorted_token_freq_pairs))
    

    vocab = {token: idx+PAD_IDX+1 for (idx, (token, freq)) in enumerate(sorted_token_freq_pairs[:end])}  # PAD_IDX is reserved for padding
    vocab[CLS_TOKEN] = len(vocab) + PAD_IDX
    # vocab[SEP_TOKEN] = len(vocab) + PAD_IDX

    print(f'Vocabulary size: {len(vocab)}')
    return vocab


class JobPostingDataset(Dataset):
    """A Dataset to be used by a data loader.
    See https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """
    def __init__(self, X_all, y_all, cls_idx, max_seq_len):
        # X_all, y_all are the labelled examples
        # cls_idx is the index of token '[CLS]' in the vocabulary
        # max_seq_len is the maximum length of a sequence allowed
        self.X_all = X_all
        self.y_all = y_all
        self.cls_idx = cls_idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.X_all)

    def __getitem__(self, idx):
        # prepend the index of the special token '[CLS]' to each sequence
        x = [self.cls_idx] + self.X_all[idx]
        # truncate a sequence if it is longer than the maximum length allowed
        if len(x) > self.max_seq_len:
            x = x[:self.max_seq_len]
        return x, self.y_all[idx]


def collate_fn(batch):
    """Merges a list of samples to form a mini-batch for model training/evaluation.
    To be used by a data loader. See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    Xb = pad_sequence([torch.tensor(x, dtype=torch.long) for (x, _) in batch], padding_value=PAD_IDX)
    yb = torch.tensor([y for (_, y) in batch], dtype=torch.float32)
    return Xb.to(DEVICE), yb.to(DEVICE)


def get_positional_encoding(emb_size, max_seq_len):
    """Compute the positional encoding.

    Args:
        emb_size (int): the dimension of positional encoding
        max_seq_len (int): the maximum allowed length of a sequence

    Returns:
        torch.tensor: positional encoding, size=(max_seq_len, emb_size)
    """
    PE = torch.zeros(max_seq_len, emb_size)

    # TODO: compute the positional encoding as specified in the question.

    # return PE
    #========================================================================
    # PE = torch.zeros(max_seq_len, emb_size)
    for pos in range(max_seq_len):
        for i in range(0, emb_size, 2):
            # For t, 2i
            PE[pos, i] = math.sin(pos / (10000 ** ((2 * i) / emb_size)))
            # For t, 2i+1
            PE[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / emb_size)))

    return PE
    #==========================================================================


class JobClassifier(nn.Module):
    """A job classifier using transformers."""
    def __init__(self, vocab_size, emb_size=128, ffn_size=128, num_tfm_layer=2, num_head=2, p_dropout=0.2, max_seq_len=300):
        """JobClassifier initialiser.
        Args:
            vocab_size (int): the size of vocabulary
            emb_size (int): the dimension of token embedding (and position encoding)
            ffn_size (int): the dimension of the feedforward network model in a transformer encoder layer
            num_tfm_layer (int): the number of transformer encoder layers
            p_dropout (float): the dropout probability (to be used in a transformer encoder layer as well as the dropout
                layer of this class.
            max_seq_len (int): the maximum allowed length of a sequence
        """
        super().__init__()
        
        self.token_embeddings = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX)
        
        # registers the positional encoding so that it is saved with the model
        # see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        self.register_buffer(
            "positional_encoding", get_positional_encoding(emb_size, max_seq_len), persistent=False
        )

        self.dropout = nn.Dropout(p=p_dropout)
        self.linear = nn.Linear(emb_size, 1)  # for binary classification

        # TODO: create a TransformerEncoder with `num_tfm_layer` TransformerEncoderLayer, see
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html

        #========================================
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_head,
            dim_feedforward=ffn_size,
            dropout=p_dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_tfm_layer)

        #===========================================
        

    def forward(self, x):
        """The forward function of SentimentClassifier.
        x: a (mini-batch) of samples, size=(SEQUENCE_LENGTH, BATCH_SIZE)
        """

        # TODO: implement the forward function as specified in the question

        # The code below needs to be modified.
        # pass
        #====================================

        # add the positional encoding to the embeddings of the input sequence
        seq_len, batch_size = x.size() # Get the size of the imput
        embeddings = self.token_embeddings(x) # Create token embeddings
        # Use only the needed part of positional encoding
        pos_encoding = self.positional_encoding[:seq_len, :].unsqueeze(1).expand(-1, batch_size, -1)
        embeddings = embeddings + pos_encoding # Add positional encoding
        
        # apply dropout
        embeddings = self.dropout(embeddings)
        
        # call the transformer encoder
        # Create padding mask
        padding_mask = (x == PAD_IDX).transpose(0, 1)
        encoded = self.transformer_encoder(embeddings, src_key_padding_mask=padding_mask)
        
        # extract the final hidden state of the [CLS] token from the transformer encoder
        cls_token_state = encoded[0, :, :]  # [CLS] token is at position 0
        
        # use the linear layer to compute the logits
        logits = self.linear(cls_token_state)
        
        return logits.squeeze(-1)
        #===========================================


def eval_model(model, dataset, batch_size=64):
    """Evaluate a trained SentimentClassifier.

    Args:
        model (JobClassifier): a trained model
        dataset (JobPostingDataset): a dataset of samples
        batch_size (int): the batch_size

    Returns:
        float: The accuracy of the model on the provided dataset
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        preds = []
        targets = []
        for (Xb, yb) in tqdm(dataloader):
            out = model(Xb)
            preds.append(out.cpu().numpy() > 0)
            targets.append(yb.cpu().numpy())
        score = accuracy_score(np.concatenate(targets), np.concatenate(preds).astype(np.int32))
    model.train()
    return score


def train_model(model, dataset_train, dataset_val, batch_size=64, num_epoch=1, learning_rate=0.001, fmodel='best_model.pth'):
    """Train a SentimentClassifier.

    Args:
        model (JobClassifier): a model to be trained
        dataset_train (JobPostingDataset): a dataset of samples (training set)
        dataset_val (JobPostingDataset): a dataset of samples (validation set)
        batch_size (int): the batch_size
        num_epoch (int): the number of training epochs
        learning_rate (float): the learning rate
        fmodel (str): name of file to save the model that achieves the best accuracy on the validation set

    Returns:
        JobClassifier: the trained model
    """
    model.train()

    # The loss function and optimiser
    loss_fn = nn.BCEWithLogitsLoss()  # the binary cross entropy loss using logits
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_batch = (len(X_train) - 1) // batch_size + 1
    print(f'{"Epoch":>10} {"Batch":>10} {"Train loss (running avg.)":>20}')
    
    # TODO: train the model for `num_epoch` epochs using the training set
    # evaluate the model on the validation set after each epoch of training
    # save the model that achieves the best accuracy on the validation set
    # see https://pytorch.org/tutorials/beginner/saving_loading_models.html

    #======================================================
    # Data loader for training set
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    best_val_accuracy = 0.0
    
    for epoch in range(num_epoch):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (Xb, yb) in enumerate(tqdm(train_dataloader)):
            optimiser.zero_grad()
            # Forward (implicit invocation)
            # In PyTorch: use a model instance like a function, 
            # it automatically calls the forward method of that model
            outputs = model(Xb)

            loss = loss_fn(outputs, yb)

            # Backward
            # PyTorch build and use the computation graph to compute gradients
            # And the gradients are stored in the .grad attribute of the model's parameters
            loss.backward()

            # Update weights
            optimiser.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f'Epoch {epoch+1:>10} Batch {batch_idx+1:>10} Train loss (running avg.): {avg_loss:>20.4f}')
        
        # Evaluate on validation set
        val_accuracy = eval_model(model, dataset_val, batch_size=batch_size)
        print(f'Validation accuracy: {val_accuracy:.4f}')
        
        # Update the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), fmodel)
            print(f'New best model saved to {fmodel}')
    
    # Load the model
    model.load_state_dict(torch.load(fmodel))
    return model
    #======================================================================


    return model


if __name__ == '__main__':
    torch.manual_seed(SEED)

    # TODO: replace the name of the file below with the data file
    # you have downloaded from Kaggle or with a data file that you
    # have preprocessed.    
    data_file = os.path.join("data", "fake_job_postings.csv")

    # Preprocess the dataset for the education classification task
    Xr_train, y_train, Xr_val, y_val, Xr_test, y_test = prepare_dataset(filename=data_file)


    get_tokenised_docs = lambda Xr: [tokenise_text(xr) for xr in tqdm(Xr)]
    get_token_indices = lambda Xt, vocab: [[vocab[token] for token in xt if token in vocab] for xt in Xt]

    Xt_train, Xt_val, Xt_test = [get_tokenised_docs(Xr) for Xr in [Xr_train, Xr_val, Xr_test]]
    vocab = build_vocab(Xt_train + Xt_val, min_freq=5)
    X_train, X_val, X_test = [get_token_indices(Xt, vocab) for Xt in [Xt_train, Xt_val, Xt_test]]

    max_seq_len = 500
    cls_idx = vocab[CLS_TOKEN]
    dataset_train = JobPostingDataset(X_train, y_train, cls_idx, max_seq_len)
    dataset_val = JobPostingDataset(X_val, y_val, cls_idx, max_seq_len)
    dataset_test = JobPostingDataset(X_test, y_test, cls_idx, max_seq_len)

    # Note that we do not directly use the combined training set and validation set to re-train the model
    # we use this strategy for simplicity, 
    # see Section 7.8 in the deep learning textbook for other possible options
    # https://www.deeplearningbook.org/contents/regularization.html

    clf = JobClassifier(
        len(vocab),
        emb_size = 300,
        ffn_size = 512,
        num_tfm_layer = 2,
        num_head = 4,
        p_dropout = 0.5,
        max_seq_len = max_seq_len,
    ).to(DEVICE)

    fmodel = 'best_model.pth'
    clf = train_model(clf, dataset_train, dataset_val, batch_size=160, num_epoch=100, learning_rate=3e-4,
                      fmodel=fmodel)

    # uncomment the code below to test the trained model

    # Print the model structure
    print(f'Loading model from {fmodel} ...')
    clf.load_state_dict(torch.load(fmodel, map_location=torch.device(DEVICE)))
    clf = clf.to(DEVICE)
    print(clf)

    acc_test = eval_model(clf, dataset_test, batch_size=256)
    print(f'Accuracy (test): {acc_test:.4f}')