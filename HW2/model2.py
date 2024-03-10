from torch import nn
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from data_load import NERDataSet


TRAIN_PATH = "data/train.tagged"
DEV_PATH = "data/dev.tagged"
EMBEDDING_PATH = ['word2vec-google-news-300']


class NERNN(nn.Module):
    def __init__(self, vec_dim, num_classes, hidden_dim=100):
        super(NERNN, self).__init__()
        self.first_layer = nn.Linear(vec_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        x = self.first_layer(input_ids)
        x = self.activation(x)
        x = self.second_layer(x)
        if labels is None:
            return x, None
        loss = self.loss(x, labels)
        return x, loss


def train(model, data_sets, optimizer, num_epochs: int, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
                    "dev": DataLoader(data_sets["dev"], batch_size=batch_size, shuffle=False)}
    model.to(device)
    phases = ['train', 'dev']

    best_f1 = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            labels, preds = [], []
            for batch in data_loaders[phase]:
                batch_size = 0
                for k, v in batch.items():
                    batch[k] = v.to(device)
                    batch_size = v.shape[0]

                optimizer.zero_grad()
                if phase == 'train':
                    outputs, loss = model(**batch)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs, loss = model(**batch)
                pred = outputs.argmax(dim=-1).clone().detach().cpu()
                labels += batch['labels'].cpu().view(-1).tolist()
                preds += pred.view(-1).tolist()
                running_loss += loss.item() * batch_size

            epoch_loss = running_loss / len(data_sets[phase])
            epoch_f1 = round(f1_score(labels, preds), 5)
            if phase.title() == "dev":
                print(f'{phase.title()} Loss: {epoch_loss:.4e} f1: {epoch_f1}')
            else:
                print(f'{phase.title()} Loss: {epoch_loss:.4e} f1: {epoch_f1}')
            if phase == 'dev' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
        print()

    print(f'Best Dev F1: {best_f1:4f}\n')


train_ds = NERDataSet(TRAIN_PATH, EMBEDDING_PATH)
print('created train')
dev_ds = NERDataSet(DEV_PATH, EMBEDDING_PATH)
datasets = {"train": train_ds, "dev": dev_ds}
nn_model = NERNN(num_classes=2, vec_dim=train_ds.vector_dim)
optimizer = Adam(params=nn_model.parameters())
train(model=nn_model, data_sets=datasets, optimizer=optimizer, num_epochs=5)
