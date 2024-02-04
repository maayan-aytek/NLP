from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple, bool


WORD = 0
TAG = 1


def search_numbers_in_word(word:str) -> Tuple[bool, bool]:
    """
    Checks if there are numeric parts in a string (word).
    Args:
        word (str): _description_

    Returns:
        Tuple[bool, bool]: first element- True if all the string is numeric, 
                           second element- True if only part of the string is numric
    """
    is_all_numeric = False
    is_combined_numeric = False
    try:
        float(word)
        is_all_numeric = True
        return is_all_numeric, is_combined_numeric
    except ValueError:
        pass
    for c in word: 
        if c.isdigit():
            is_combined_numeric = True
            return is_all_numeric, is_combined_numeric
    return is_all_numeric, is_combined_numeric


class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        feature_dict_list = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107"]  # the feature classes used in the code
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}
        '''
        A dictionary containing the counts of each data regarding a feature class. For example in f100, would contain
        the number of times each (word, tag) pair appeared in the text.
        '''
        self.tags = set()  # a set of all the seen tags
        self.tags.add("~")
        self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the test

    def get_word_tag_pair_count(self, file_path) -> None:
        """
            Extract out of text all word/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        with open(file_path) as file:
            for line in file:
                if line[-1:] == "\n":
                    line = line[:-1]
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    self.tags.add(cur_tag)
                    # calculating maximum cut size for prefix and suffix
                    word_cutting_bound = min(len(cur_word) + 1, 5)
                    # count seperatly words and count 
                    self.tags_counts[cur_tag] += 1
                    self.words_count[cur_word] += 1
                    
                    # count repetitions of word-tags pairs 
                    if (cur_word, cur_tag) not in self.feature_rep_dict["f100"]:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] += 1
                    
                    for suffix_len in range(1, word_cutting_bound):
                        if (cur_word[-suffix_len:], cur_tag) not in self.feature_rep_dict["f101"]:
                            self.feature_rep_dict["f101"][(cur_word[-suffix_len:], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f101"][(cur_word[-suffix_len:], cur_tag)] += 1
                    
                    for prefix_len in range(1, word_cutting_bound):
                        if (cur_word[:prefix_len], cur_tag) not in self.feature_rep_dict["f102"]:
                            self.feature_rep_dict["f102"][(cur_word[:prefix_len], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f102"][(cur_word[:prefix_len], cur_tag)] += 1

                # w[-2] = w[-1] = "*"
                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                # end of sentence: w[n+1] = "~"
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    # history = (w[current], t[current], w[-1], t[-1], w[-2], t[-2], w[+1])
                    cur_word, cur_tag = sentence[i][WORD], sentence[i][TAG]
                    prev_word, prev_tag = sentence[i - 1][WORD], sentence[i - 1][TAG]
                    prev_prev_word, prev_prev_tag = sentence[i - 2][WORD], sentence[i - 2][TAG]
                    next_word = sentence[i + 1][WORD]

                    history = (
                        cur_word,cur_tag, prev_word, prev_tag, prev_prev_word,
                        prev_prev_tag, next_word)
                    
                    # f103-f107
                    if (prev_prev_tag, prev_tag, cur_tag) not in self.feature_rep_dict["f103"]:
                        self.feature_rep_dict["f103"][(prev_prev_tag, prev_tag, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f103"][(prev_prev_tag, prev_tag, cur_tag)] += 1

                    if (prev_tag, cur_tag) not in self.feature_rep_dict["f104"]:
                        self.feature_rep_dict["f104"][(prev_tag, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f104"][(prev_tag, cur_tag)] += 1

                    if cur_tag not in self.feature_rep_dict["f105"]:
                        self.feature_rep_dict["f105"][cur_tag] = 1
                    else:
                        self.feature_rep_dict["f105"][cur_tag] += 1

                    if (prev_word, cur_tag) not in self.feature_rep_dict["f106"]:
                        self.feature_rep_dict["f106"][(prev_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f106"][(prev_word, cur_tag)] += 1

                    if (next_word, cur_tag) not in self.feature_rep_dict["f107"]:
                        self.feature_rep_dict["f107"][(next_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f107"][(next_word, cur_tag)] += 1

                    self.histories.append(history)


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.feature_to_idx = {
            "f100": OrderedDict(),
            "f101": OrderedDict(),
            "f102": OrderedDict(),
            "f103": OrderedDict(),
            "f104": OrderedDict(),
            "f105": OrderedDict(),
            "f106": OrderedDict(),
            "f107": OrderedDict(),
        }
        self.represent_input_with_features = OrderedDict()
        self.histories_matrix = OrderedDict()
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_idx
        """
        for feat_class in self.feature_statistics.feature_rep_dict:
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        """
        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []
        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for r, y_tag in enumerate(self.feature_statistics.tags):
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                            shape=(len(self.feature_statistics.tags) * len(
                                                self.feature_statistics.histories), self.n_total_features),
                                            dtype=bool)
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(
                self.feature_statistics.histories), self.n_total_features), dtype=bool)


def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]])\
        -> List[int]:
    """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all features that are relevant to the given history
    """
    c_word = history[0]
    c_tag = history[1]
    prev_word = history[2]
    prev_tag = history[3]
    prev_prev_tag = history[5]
    next_word = history[6]
    features = []

    # f100
    if (c_word, c_tag) in dict_of_dicts["f100"]:
        features.append(dict_of_dicts["f100"][(c_word, c_tag)])

    # calculating maximum cut size for prefix and suffix
    # f101
    word_cutting_bound = min(len(c_word) + 1, 5)
    for suffix_len in range(1, min(word_cutting_bound + 1, 5)):
        word_suffix = c_word[-suffix_len:]
    if (word_suffix, c_tag) in dict_of_dicts["f101"]:
        features.append(dict_of_dicts["f101"][(word_suffix, c_tag)])

    # f102
    for prefix_len in range(1, min(word_cutting_bound + 1, 5)):
        word_prefix = c_word[:prefix_len]
    if (word_prefix, c_tag) in dict_of_dicts["f102"]:
        features.append(dict_of_dicts["f102"][(word_prefix, c_tag)])

    # f103
    if (prev_prev_tag, prev_tag, c_tag) in dict_of_dicts["f103"]:
        features.append(dict_of_dicts["f103"][(prev_prev_tag, prev_tag,c_tag)])

    # f104
    if (prev_tag,c_tag) in dict_of_dicts["f104"]:
        features.append(dict_of_dicts["f104"][(prev_tag,c_tag)])

    # f105
    if c_tag in dict_of_dicts["f105"]:
        features.append(dict_of_dicts["f105"][c_tag])

    # f106
    if (prev_word, c_tag) in dict_of_dicts["f106"]:
        features.append(dict_of_dicts["f106"][(prev_word, c_tag)])

    # f107
    if (next_word, c_tag) in dict_of_dicts["f107"]:
        features.append(dict_of_dicts["f107"][(next_word, c_tag)])

    return features


def preprocess_train(train_path, threshold):
    # Statistics
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    # feature2id
    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()
    print(feature2id.n_total_features)

    for dict_key in feature2id.feature_to_idx:
        print(dict_key, len(feature2id.feature_to_idx[dict_key]))
    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences
