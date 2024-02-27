from gensim import downloader
import numpy as np
import torch
import re
from torch.utils.data import Dataset
import os
import pickle


def data_loader(file_path: str, embedding_path: str, sentences_representation: bool = False):
    if sentences_representation:
        if os.path.exists(f'{file_path}_sentences.pickle') and os.path.exists(f'{file_path}_sentences_labels.pickle') and os.path.exists(f'{file_path}_words_labels.pickle'):
            sentences = pickle.load(open(f'{file_path}_sentences.pickle', 'rb'))
            sentences_labels = pickle.load(open(f'{file_path}_sentences_labels.pickle', 'rb'))
            words_labels = pickle.load(open(f'{file_path}_words_labels.pickle', 'rb'))
            return sentences, sentences_labels, words_labels
    else:
        if os.path.exists(f'{file_path}_embedded_words.pickle') and os.path.exists(f'{file_path}_words_labels.pickle'):
            embedded_words = pickle.load(f'{file_path}_embedded_words.pickle')
            words_labels = pickle.load(f'{file_path}_words_labels.pickle')
            return sentences, sentences_labels, words_labels

    words, embedded_words = [], []
    sentences_labels = []
    words_labels = []
    sentences = []
    curr_sentence = []
    curr_sentence_labels = []
    embedding_model = downloader.load(embedding_path)
    embedding_dim = len(embedding_model[list(embedding_model.key_to_index.keys())[0]])
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line == "\n" or line == "\t\n":
            sentences.append(curr_sentence)
            sentences_labels.append(curr_sentence_labels)
            curr_sentence = []
            curr_sentence_labels = []
        else:
            splitted_line = line.split('\t')
            if len(splitted_line) != 2:
                print(f"Illegal line {i}: {line}")
                continue
            curr_word, curr_label = splitted_line
            coded_label = 0 if curr_label[:-1] == 'O' else 1
            if curr_word in embedding_model: 
                word_vec = embedding_model[curr_word]
            elif curr_word.lower() in embedding_model:
                word_vec = embedding_model[curr_word.lower()]
            elif curr_word.capitalize() in embedding_model:
                word_vec = embedding_model[curr_word.capitalize()]
            elif "https" in curr_word and "https" in embedding_model:
                word_vec = embedding_model["https"]
            elif "http" in curr_word and "http" in embedding_model:
                word_vec = embedding_model["http"]
            else: 
                cleaned_word = re.sub(r'[^a-zA-Z]', '', curr_word)
                if cleaned_word in embedding_model: 
                    word_vec = embedding_model[cleaned_word]
                elif cleaned_word.lower() in embedding_model:
                    word_vec = embedding_model[cleaned_word.lower()]
                elif cleaned_word.capitalize() in embedding_model:
                    word_vec = embedding_model[cleaned_word.capitalize()]
                else:
                    word_vec = np.zeros(embedding_dim)
            curr_sentence_labels.append(coded_label)
            curr_sentence.append(word_vec)
            words.append(curr_word)
            embedded_words.append(word_vec)
            words_labels.append(coded_label)
    
    with open(f'{file_path}_words_labels.pickle', 'wb') as file:
            pickle.dump(words_labels, file)
    if sentences_representation:
        with open(f'{file_path}_sentences.pickle', 'wb') as file:
            pickle.dump(sentences, file)
        with open(f'{file_path}_sentences_labels.pickle', 'wb') as file:
            pickle.dump(sentences_labels, file)
        return sentences, sentences_labels, words_labels
    else:
        with open(f'{file_path}_embedded_words.pickle', 'wb') as file:
            pickle.dump(embedded_words, file)
        return embedded_words, words_labels


class NERDataSet(Dataset):
    def __init__(self, file_path: str, embedding_path: str, sentences_representation: bool = False):
        if sentences_representation:
            sentences, sentences_labels, words_labels = data_loader(file_path, embedding_path, sentences_representation)
            tensor_sentences = []
            for sentence in sentences:
                tensor_word_vecs = []
                for word_vec in sentence:
                    tensor_word_vecs.append(torch.tensor(word_vec))
                tensor_sentences.append(torch.stack(tensor_word_vecs))
            # print(file_path, sum(words_labels)/len(words_labels))
            self.labels = [torch.tensor(labels_list) for labels_list in sentences_labels]
            self.tags_to_idx = {tag: idx for idx, tag in enumerate(sorted(list(set(words_labels))))}
            self.idx_to_tag = {idx: tag for tag, idx in self.tags_to_idx.items()}
            self.tokenized_sen = tensor_sentences
            self.vector_dim = tensor_sentences[0].shape[-1]
        else:
            embedded_words, words_labels = data_loader(file_path, embedding_path)
            self.labels = words_labels
            self.tags_to_idx = {tag: idx for idx, tag in enumerate(sorted(list(set(self.labels))))}
            self.idx_to_tag = {idx: tag for tag, idx in self.tags_to_idx.items()}
            embedded_words = np.stack(embedded_words)
            self.tokenized_sen = embedded_words
            self.vector_dim = embedded_words.shape[-1]

    def __getitem__(self, item): # will caled only when sentences_representation = False
        cur_sen = self.tokenized_sen[item]
        cur_sen = torch.FloatTensor(cur_sen).squeeze()
        label = self.labels[item]
        label = self.tags_to_idx[label]
        data = {"input_ids": cur_sen, "labels": label}
        return data

    def __len__(self):
        return len(self.labels)
