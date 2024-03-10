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
EMBEDDING_PATH = ['word2vec-google-news-300']


class NER_LSTM(nn.Module):
    def __init__(self, vec_dim, num_classes, hidden_dim=100):
        super(NER_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.tag_dim = num_classes
        self.vec_dim = vec_dim
        self.lstm = nn.LSTM(self.vec_dim, self.hidden_dim)
        self.hidden2tag = nn.Sequential(nn.ReLU(),
                                        nn.Linear(self.hidden_dim, self.tag_dim))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        lstm_out, _ = self.lstm(input_ids.unsqueeze(1).float())
        tag_space = self.hidden2tag(lstm_out)
        tag_score =  F.softmax(tag_space.squeeze(1), dim=1)
        if labels is None:
            return tag_score, None
        loss = self.loss_fn(tag_score, labels)
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
            for sentence, sentence_labels in zip(dataset.tokenized_sen, dataset.labels):
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
        print()
    print(f'Best Dev F1: {best_f1:4f}\n')

train_ds = NERDataSet(TRAIN_PATH, EMBEDDING_PATH, sentences_representation=True)
print('created train')
dev_ds = NERDataSet(DEV_PATH, EMBEDDING_PATH, sentences_representation=True)
datasets = {"train": train_ds, "dev": dev_ds}
nn_model = NER_LSTM(num_classes=2, vec_dim=train_ds.vector_dim)
optimizer = Adam(params=nn_model.parameters())
train(model=nn_model, data_sets=datasets, optimizer=optimizer, num_epochs=5)
