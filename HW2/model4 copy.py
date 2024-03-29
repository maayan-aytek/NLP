from torch import nn
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from data_load import NERDataSet
import torch.nn.functional as F


TRAIN_PATH = "data/train.tagged"
DEV_PATH = "data/dev.tagged"
EMBEDDING_PATH = ['word2vec-google-news-300','glove-twitter-50']# ['word2vec-google-news-300', 'glove-twitter-50'] 
# ['word2vec-google-news-300', 'glove-twitter-50', 'glove-wiki-gigaword-50'] # "glove-twitter-50" # "conceptnet-numberbatch-17-06-300" # "glove-twitter-50" "glove-twitter-100"  


class NER_LSTM(nn.Module):
    def __init__(self, vec_dim, num_classes, device, hidden_dim=128):
        super(NER_LSTM, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.tag_dim = num_classes
        self.vec_dim = vec_dim
        self.lstm = nn.LSTM(self.vec_dim, self.hidden_dim, bidirectional=True)
        self.hidden2tag = self.hidden2tag = nn.Sequential(nn.Tanh(),
                                                        nn.Linear(2 * self.hidden_dim, self.tag_dim))
        self.loss_fn = nn.CrossEntropyLoss(torch.tensor([1.0, 10.0]))

    def forward(self, input_ids, labels=None):
        lstm_out, _ = self.lstm(input_ids.unsqueeze(1).float().to(self.device))
        tag_space = self.hidden2tag(lstm_out)
        tag_score =  F.softmax(tag_space.squeeze(1), dim=1)
        if labels is None:
            return tag_score, None
        loss = self.loss_fn(tag_score.to(self.device), labels.to(self.device))
        return tag_score, loss


def train(model, data_sets, optimizer, num_epochs: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model.to(device)
    phases = ["train", "dev"]
    best_f1 = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} of {num_epochs}")
        print("-" * 10)
        for phase in phases:
            if phase == "train":
                model.train()
            else:
                model.eval()

            labels, preds = [], []
            dataset = data_sets[phase]
            for i, (sentence, sentence_labels) in enumerate(zip(dataset.tokenized_sen, dataset.labels)):
                if phase == "train":
                    outputs, loss = model(sentence, sentence_labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    with torch.no_grad():
                        outputs, loss = model(sentence, sentence_labels)

                pred = outputs.argmax(dim=1).clone().detach().cpu()
                labels += sentence_labels.cpu().tolist()
                preds += pred.tolist()

            epoch_f1 = f1_score(labels, preds)
            print(f"{phase} F1: {epoch_f1}")

            if phase == "dev" and epoch_f1 > best_f1:
                best_f1 = epoch_f1
    return best_f1

import time 

t0 = time.time()
train_ds = NERDataSet(TRAIN_PATH, EMBEDDING_PATH, sentences_representation=True)
print('created train')
dev_ds = NERDataSet(DEV_PATH, EMBEDDING_PATH, sentences_representation=True)
datasets = {"train": train_ds, "dev": dev_ds}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_model = NER_LSTM(num_classes=2, vec_dim=train_ds.vector_dim, device=device)
optimizer = Adam(params=nn_model.parameters())
train(model=nn_model, data_sets=datasets, optimizer=optimizer, num_epochs=8)
print(time.time() - t0)

def find_optinal_embedding(embedding_options_list):
    import itertools
    import pickle
    groups = list(itertools.combinations(embedding_options_list, 2)) + list(itertools.combinations(embedding_options_list, 3))
    groups_lst = []
    for elem in groups:
        groups_lst.append(list(elem))
    
    best_f1 = 0
    best_f1_comb = ""
    result_dict = {}
    for embedding_combination in groups_lst:
        train_ds = NERDataSet(TRAIN_PATH, embedding_combination, sentences_representation=True)
        dev_ds = NERDataSet(DEV_PATH, embedding_combination, sentences_representation=True)
        datasets = {"train": train_ds, "dev": dev_ds}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn_model = NER_LSTM(num_classes=2, vec_dim=train_ds.vector_dim, device=device)
        optimizer = Adam(params=nn_model.parameters())
        f1 = train(model=nn_model, data_sets=datasets, optimizer=optimizer, num_epochs=15)
        result_dict[tuple(embedding_combination)] = f1
        if f1 > best_f1:
            best_f1 = f1
            best_f1_comb = embedding_combination
        print(f"{embedding_combination}: {f1}")
    
    print(f"Best result is {best_f1} by {best_f1_comb}")
    with open(f'embedding_combs_f1_dict.pickle', 'wb') as file:
            pickle.dump(result_dict, file)

# find_optinal_embedding(['word2vec-google-news-300', 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200',
#                          'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300'])


