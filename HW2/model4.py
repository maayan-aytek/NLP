from torch import nn
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import Dataset
from data_load import NERDataSet
import torch.nn.functional as F
import numpy as np


TRAIN_PATH = "data/train.tagged"
DEV_PATH = "data/dev.tagged"
TEST_PATH = "data/test.untagged"
SMALL_EMBEDDING_PATH = ['word2vec-google-news-300', 'glove-twitter-50']
LARGE_EMBEDDING_PATH = ['word2vec-google-news-300', 'glove-twitter-200', 'glove-wiki-gigaword-200']

class NER_LSTM(nn.Module):
    def __init__(self, vec_dim, device, hidden_dim, positive_class_weight, drop_out_list, activations, num_classes=2):
        super(NER_LSTM, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.tag_dim = num_classes
        self.vec_dim = vec_dim
        self.lstm = nn.LSTM(self.vec_dim, self.hidden_dim, bidirectional=True)
        if len(activations) == 1: # One activation function indicates one linear layer and one dropout.
            self.hidden2tag = nn.Sequential(
                                        nn.Dropout(drop_out_list[0]),
                                        activations[0],
                                        nn.Linear(2 * self.hidden_dim, self.tag_dim))
        if len(activations) == 2: # Two activation functions indicate two linear layers and two dropouts.
            self.hidden2tag = nn.Sequential(nn.Dropout(drop_out_list[0]),
                                            activations[0],
                                            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                                            nn.Dropout(drop_out_list[1]),
                                            activations[1],
                                            nn.Linear(self.hidden_dim, self.tag_dim))
        self.loss_fn = nn.CrossEntropyLoss(torch.tensor([1.0, positive_class_weight])) # Weighted loss to punish more if model doesn't recognize entities.


    def forward(self, input_ids, labels=None):
        lstm_out, _ = self.lstm(input_ids.unsqueeze(1).float().to(self.device)) # Unsqueez to convert tensor's shape to 3D
        tag_space = self.hidden2tag(lstm_out) # Feed Forward network
        tag_score =  F.softmax(tag_space.squeeze(1), dim=1) # Transform to probabilities.
        if labels is None: # Test
            return tag_score
        loss = self.loss_fn(tag_score.to(self.device), labels.to(self.device)) # Calculate loss in comparission with true labels.
        return tag_score, loss


def train(model, data_sets, optimizer, num_epochs: int, test_mode=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model.to(device)
    phases = ["train", "test"] if test_mode else ["train", "dev"]
    best_f1 = 0
    best_epoch = 0
    best_preds = []
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
            if phase != "test":
                for sentence, sentence_labels in zip(dataset.tokenized_sen, dataset.labels): # Iterate on sentences for context and tag words.
                    if phase == "train":
                        outputs, loss = model(sentence, sentence_labels)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    else: # dev
                        with torch.no_grad():
                            outputs, loss = model(sentence, sentence_labels)

                    pred = outputs.argmax(dim=1).clone().detach().cpu()
                    labels += sentence_labels.cpu().tolist()
                    preds += pred.tolist()

                epoch_f1 = f1_score(labels, preds)
                print(f"{phase} F1: {epoch_f1}")
                if phase == "dev" and epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_preds = preds
                    best_epoch = epoch

            else: # Test- has no labels
                for sentence in dataset.tokenized_sen: # Iterate on sentences for context and tag words on test.
                    with torch.no_grad():
                        outputs = model(sentence)

                    pred = outputs.argmax(dim=1).clone().detach().cpu()
                    preds += pred.tolist()
    if test_mode:
        return preds
    return best_f1, np.array(best_preds), best_epoch


def flatten_list_of_tensors(list_of_tensors): # Convert list of tensor elements to a flat list. 
    flattened_list = []
    for tensor in list_of_tensors:
        flattened_list.extend(tensor.flatten().tolist())
    return flattened_list


def train_and_tag(test_mode, num_epoch_list, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    small_train_ds = NERDataSet(TRAIN_PATH, SMALL_EMBEDDING_PATH, sentences_representation=True)
    large_train_ds = NERDataSet(TRAIN_PATH, LARGE_EMBEDDING_PATH, sentences_representation=True)
    if not test_mode:
        small_dev_ds = NERDataSet(DEV_PATH, SMALL_EMBEDDING_PATH, sentences_representation=True)
        large_dev_ds = NERDataSet(DEV_PATH, LARGE_EMBEDDING_PATH, sentences_representation=True)
        small_datasets = {"train": small_train_ds, "dev": small_dev_ds}
        large_datasets = {"train": large_train_ds, "dev": large_dev_ds}
    else:
        small_test_ds = NERDataSet(TEST_PATH, SMALL_EMBEDDING_PATH, sentences_representation=True)
        large_test_ds = NERDataSet(TEST_PATH, LARGE_EMBEDDING_PATH, sentences_representation=True)
        small_datasets = {"train": small_train_ds, "dev": small_test_ds}
        large_datasets = {"train": large_train_ds, "dev": large_test_ds}
    

    nn_model1 = NER_LSTM(vec_dim=large_train_ds.vector_dim, device=device, hidden_dim=100, positive_class_weight=15, drop_out_list=[0.2, 0.2], activations=[nn.Tanh(), nn.ReLU()]) 
    nn_model2 = NER_LSTM(vec_dim=large_train_ds.vector_dim, device=device, hidden_dim=64, positive_class_weight=15, drop_out_list=[0.1], activations=[nn.ReLU()]) 
    nn_model3 = NER_LSTM(vec_dim=small_train_ds.vector_dim, device=device, hidden_dim=100, positive_class_weight=19, drop_out_list=[0.0], activations=[nn.ReLU()]) 
    nn_model4 = NER_LSTM(vec_dim=small_train_ds.vector_dim, device=device, hidden_dim=100, positive_class_weight=19, drop_out_list=[0.2, 0], activations=[nn.Tanh(), nn.ReLU()])
    nn_model5 = NER_LSTM(vec_dim=small_train_ds.vector_dim, device=device, hidden_dim=128, positive_class_weight=19, drop_out_list=[0.2], activations=[nn.Tanh()])

    optimizer1 = Adam(params=nn_model1.parameters())
    optimizer2 = Adam(params=nn_model2.parameters())
    optimizer3 = Adam(params=nn_model3.parameters())
    optimizer4 = Adam(params=nn_model4.parameters())
    optimizer5 = Adam(params=nn_model5.parameters())

    if not test_mode:
        best_f1_model1, best_preds1, best_epoch1 = train(model=nn_model1, data_sets=large_datasets, optimizer=optimizer1, num_epochs=num_epoch_list[0], test_mode=test_mode)
        best_f1_model2, best_preds2, best_epoch2 = train(model=nn_model2, data_sets=large_datasets, optimizer=optimizer2, num_epochs=num_epoch_list[1], test_mode=test_mode)
        best_f1_model3, best_preds3, best_epoch3 = train(model=nn_model3, data_sets=small_datasets, optimizer=optimizer3, num_epochs=num_epoch_list[2], test_mode=test_mode)
        best_f1_model4, best_preds4, best_epoch4 = train(model=nn_model4, data_sets=small_datasets, optimizer=optimizer4, num_epochs=num_epoch_list[3], test_mode=test_mode)
        best_f1_model5, best_preds5, best_epoch5 = train(model=nn_model5, data_sets=small_datasets, optimizer=optimizer5, num_epochs=num_epoch_list[4], test_mode=test_mode)

        final_preds = np.floor((best_preds1 + best_preds2 + best_preds3 + best_preds4 + best_preds5) / 3).tolist() # Label as majority's vote
        dev_labels = flatten_list_of_tensors(large_dev_ds.labels)
        print(f"model1 f1 in epoch {best_epoch1}, seed {seed}:", best_f1_model1)
        print(f"model2 f1 in epoch {best_epoch2}, seed {seed}:", best_f1_model2)
        print(f"model3 f1 in epoch {best_epoch3}, seed {seed}:", best_f1_model3)
        print(f"model4 f1 in epoch {best_epoch4}, seed {seed}:", best_f1_model4)
        print(f"model5 f1 in epoch {best_epoch5}, seed {seed}:", best_f1_model5)
        print(f"\nFinal f1 seed {seed}:", f1_score(dev_labels, final_preds))

    else:
        preds1= train(model=nn_model1, data_sets=large_datasets, optimizer=optimizer1, num_epochs=num_epoch_list[0], test_mode=test_mode)
        preds2 = train(model=nn_model2, data_sets=large_datasets, optimizer=optimizer2, num_epochs=num_epoch_list[1], test_mode=test_mode)
        preds3 = train(model=nn_model3, data_sets=small_datasets, optimizer=optimizer3, num_epochs=num_epoch_list[2], test_mode=test_mode)
        preds4 = train(model=nn_model4, data_sets=small_datasets, optimizer=optimizer4, num_epochs=num_epoch_list[3], test_mode=test_mode)
        preds5 = train(model=nn_model5, data_sets=small_datasets, optimizer=optimizer5, num_epochs=num_epoch_list[4], test_mode=test_mode)

        final_preds = np.floor((preds1 + preds2 + preds3 + preds4 + preds5) / 3).tolist() # Label as majority's vote
    
    return final_preds


def generate_preds_file(preds):
    preds_file_name = "comp_206713612_316111442.tagged"
    with open(preds_file_name, 'w', encoding='utf-8') as write_file:
        with open(TEST_PATH, 'r', encoding='utf-8') as read_file:
            lines = read_file.readlines()
            for i, line in enumerate(lines):
                if line == "\n" or line == "\t\n":
                    write_file.write(line)
                else:
                    curent_word = line.split('\n')[0] #TODO: check tabs
                    current_label = 'O' if preds[i] == 0 else "I"
                    word_with_pred_line = f'{curent_word}\t{current_label}\n'
                    write_file.write(word_with_pred_line)


def find_optimal_architecture(): # used for R&D- network architecture experiments
    import pickle 
    drop_out_list = [0, 0.1, 0.2]
    hidden_dims = [64, 100, 128]
    lr_list = [0.001, 0.01, 0.1]
    activations = [[nn.Tanh()], [nn.ReLU()], [nn.Tanh(), nn.ReLU()],
                   [nn.ReLU(), nn.Tanh()], [nn.Tanh(), nn.Tanh()], [nn.ReLU(), nn.ReLU()]]
    minor_class_weight = [6.0, 10.0, 15.0, 19.0]
    embeddings = [['word2vec-google-news-300','glove-twitter-200', 'glove-wiki-gigaword-200'], ['word2vec-google-news-300', 'glove-twitter-50']]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_f1 = 0
    best_f1_comb = ""
    result_dict = {}
    
    count = 0
    for embedding in embeddings:
        train_ds = NERDataSet(TRAIN_PATH, embedding, sentences_representation=True)
        dev_ds = NERDataSet(DEV_PATH, embedding, sentences_representation=True)
        datasets = {"train": train_ds, "dev": dev_ds}
        for lr in lr_list:
            for weight in minor_class_weight:
                for hidden_dim in hidden_dims:
                    for activation_comb in activations:
                        drop_out_options = []
                        if len(activation_comb) == 1:
                            for drop_out1 in drop_out_list:
                                drop_out_options.append([drop_out1])
                        else:
                            for drop_out1 in drop_out_list:
                                for drop_out2 in drop_out_list:
                                    if [drop_out1, drop_out2] not in drop_out_options:
                                        drop_out_options.append([drop_out1, drop_out2])
                        for drop_out in drop_out_options:
                            if count > 1196:
                                nn_model = NER_LSTM(num_classes=2, vec_dim=train_ds.vector_dim, device=device,
                                        drop_out_list=drop_out, activations=activation_comb, minor_class_weight=weight, hidden_dim=hidden_dim)
                                optimizer = Adam(params=nn_model.parameters(), lr=lr)
                                f1 = train(model=nn_model, data_sets=datasets, optimizer=optimizer, num_epochs=8)
                                result_dict[(tuple(embedding), tuple(drop_out), lr, tuple(activation_comb), weight, hidden_dim)] = f1
                                if f1 > best_f1:
                                    best_f1 = f1
                                    best_f1_comb = (tuple(embedding), tuple(drop_out), lr, tuple(activation_comb), weight, hidden_dim)
                                print(f"comb {count}: {(tuple(embedding), tuple(drop_out), lr, tuple(activation_comb), weight, hidden_dim)}: {f1}")
                            count += 1
                                
    print(f"Best result is {best_f1} by {best_f1_comb}")
    with open(f'architecture_combs_f1_dict.pickle', 'wb') as file:
            pickle.dump(result_dict, file)


seeds = [63644, 23196, 19096, 84665, 60110, 32194, 271, 63659, 95491, 8255, 38529, 56071, 16506, 1108, 1910, 42,
          48476, 55940, 14957, 73745, 4709, 2013, 93350, 61632, 98167, 87306, 18035, 91267, 36041, 38140]
for seed in seeds:
    torch.manual_seed(seed)
    train_and_tag(test_mode=False, num_epoch_list=[25, 25, 25, 25, 25], seed=seed)