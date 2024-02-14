from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple


WORD = 0
TAG = 1



def search_numbers_in_word(word: str) -> Tuple[bool, bool]:
    """
    Checks if there are numeric parts in a string (word).
    Args:
        word (str): string word

    Returns:
        Tuple[bool, bool]: first element- True if all the string is numeric, 
                           second element- True if only part of the string is numeric
    """
    is_all_numeric = False
    is_combined_numeric = False
    try:
        float(word.replace(',', ''))
        is_all_numeric = True
        return is_all_numeric, is_combined_numeric
    except ValueError:
        pass
    for c in word: 
        if c.isdigit():
            is_combined_numeric = True
            return is_all_numeric, is_combined_numeric
    return is_all_numeric, is_combined_numeric


def check_capital(word: str) -> Tuple[bool, bool]: 
    """Checks if the word starts with capital letters or contains only capital letters.
    Args:
        word (str): string word

    Returns:
        Tuple[bool, bool]: first element- True if the string only contains capital letters, 
                           second element- True if only the first letter is capital.
    """
    is_all_capital = False
    first_capital = False
    if word == "A" or word == "I":
        first_capital = True
        return is_all_capital, first_capital
    if word.isupper():
        is_all_capital = True
        return is_all_capital, first_capital    
    if word[0].isupper():
        first_capital = True
    return is_all_capital, first_capital


def check_mid_word_capital(word: str) -> bool:
    """Checks if there are captial letters in the middle of the word

    Args:
        word (str): string word

    Returns:
        bool: True if there is a capital letter in the middle of the word and False otherwise
    """
    if  not word.isupper() and not word[0].isupper() and len(word) > 1:
        for c in word[1:]:
            if c.isupper():
                return True
    return False


def is_preposition(word: str) -> bool:
    """Checks if a word is a proposition (from a static list)

    Args:
        word (str): string word

    Returns:
        bool: True if the word is a proposition from the list, False otherwise
    """
    prepositions = ["about", "of", "on", "at", "in", "from", "within", "since", "than", "with", "under", "across", "above", "by", "between",
                    "along", "down", "up", "ago"]
    if word in prepositions:
        return True
    return False


def is_determiner(word: str) -> bool:
    """Checks if a word is a determiner (from a static list)

    Args:
        word (str): string word

    Returns:
        bool: True if the word is a determiner from the list, False otherwise
    """
    determiners = ["a", "an", "the", "this", "that", "these", "those", "my", "yours", "is", "her", "its", "our", "their"]
    if word in determiners:
        return True
    return False


def adjective_suffixes(word: str) -> bool:
    """Checks if a word has a adjective known suffix (from a static list)

    Args:
        word (str): string word

    Returns:
        bool: True if the word has an adjective suffix from the list, False otherwise
    """
    suffixes = ["ant", "ent", "ful", "able", "is", "less", "like", "ous", "ose", "some", "ible", "ic", "ist", "y", "ive", "ent", "al", "an", "ian", "ary", "ish", "ile", "or"]
    word_len = len(word)
    for suffix in suffixes:
        suffix_len = len(suffix)
        if suffix_len <= word_len and word[-suffix_len:] == suffix:
            return True
    return False


def plural_suffixes(word: str) -> bool:
    """Checks if a word has a plural noun known suffix (from a static list)

    Args:
        word (str): string word

    Returns:
        bool: True if the word has an plural suffix from the list, False otherwise
    """
    suffixes = ["s", "es", "ies", "ves", "en"]
    word_len = len(word)
    for suffix in suffixes:
        suffix_len = len(suffix)
        if suffix_len <= word_len and word[-suffix_len:] == suffix:
            return True
    return False


def past_suffixes(word: str) -> bool:
    """Checks if a word has a past verb known suffix (from a static list)

    Args:
        word (str): string word

    Returns:
        bool: True if the word has an past suffix from the list, False otherwise
    """
    suffixes = ["ed", "d", "id", "t", "en", "ought"]
    word_len = len(word)
    for suffix in suffixes:
        suffix_len = len(suffix)
        if suffix_len <= word_len and word[-suffix_len:] == suffix:
            return True
    return False


def present_suffix(word: str) -> bool:
    """Checks if a word has a present verb known suffix - ing

    Args:
        word (str): string word

    Returns:
        bool: True if the word has an ing suffix, False otherwise
    """
    suffix = "ing"
    if len(suffix) <= len(word) and word[-len(suffix):] == suffix:
        return True
    return False


def medical_suffix(word: str) -> bool:
    """Checks if a word has medical known suffix (from a static list)- used in model 2

    Args:
        word (str): string word

    Returns:
        bool: True if the word has a medical suffix from the list, False otherwise
    """
    suffixes = ["oid", "itis", "ine", "ase", "yl", "sol"]
    word_len = len(word)
    for suffix in suffixes:
        suffix_len = len(suffix)
        if suffix_len <= word_len and word[-suffix_len:] == suffix:
            return True
    return False
    

class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        feature_dict_list = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107", "f_is_numeric", "f_is_combined_numeric", "f_all_capital", "f_first_capital",
                              "f_length", "f_next_length", "f_has_hyphen", "f_has_dot", "f_has_apostrophe",
                              "f_has_adj_suffix", "f_has_plural_suffix", "f_has_past_suffix", "f_has_present_suffix",
                              "f_mid_capital", "f106_prev_tag", "f_prev_preposition", "f_prev_determiner",
                              "f_is_preposition", "f_has_medical_suffix", "f101_5", "f101_6", "f102_5", "f102_6"]  # the feature classes used in the code
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
                    cur_word_lower = cur_word.lower()
                    cur_word_len = len(cur_word)
                    self.tags.add(cur_tag)
                    # calculating maximum cut size for prefix and suffix
                    word_cutting_bound_4 = min(len(cur_word) + 1, 5)
                    word_cutting_bound_5 = min(len(cur_word) + 1, 6)
                    word_cutting_bound_6 = min(len(cur_word) + 1, 7)
                    # count seperatly words and count 
                    self.tags_counts[cur_tag] += 1
                    self.words_count[cur_word] += 1
                    
                    # count repetitions of word-tags pairs 
                    if (cur_word, cur_tag) not in self.feature_rep_dict["f100"]:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f100"][(cur_word, cur_tag)] += 1

                    for suffix_len in range(1, word_cutting_bound_4):
                        if (cur_word[-suffix_len:], cur_tag) not in self.feature_rep_dict["f101"]:
                            self.feature_rep_dict["f101"][(cur_word[-suffix_len:], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f101"][(cur_word[-suffix_len:], cur_tag)] += 1
                    
                    for prefix_len in range(1, word_cutting_bound_4):
                        if (cur_word[:prefix_len], cur_tag) not in self.feature_rep_dict["f102"]:
                            self.feature_rep_dict["f102"][(cur_word[:prefix_len], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f102"][(cur_word[:prefix_len], cur_tag)] += 1

                    # 101-102 with max 5 suffix
                    for suffix_len in range(1, word_cutting_bound_5):
                        if (cur_word[-suffix_len:], cur_tag) not in self.feature_rep_dict["f101_5"]:
                            self.feature_rep_dict["f101_5"][(cur_word[-suffix_len:], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f101_5"][(cur_word[-suffix_len:], cur_tag)] += 1
                    
                    for prefix_len in range(1, word_cutting_bound_5):
                        if (cur_word[:prefix_len], cur_tag) not in self.feature_rep_dict["f102_5"]:
                            self.feature_rep_dict["f102_5"][(cur_word[:prefix_len], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f102_5"][(cur_word[:prefix_len], cur_tag)] += 1

                    # 101-102 with max 6 suffix
                    for suffix_len in range(1, word_cutting_bound_6):
                        if (cur_word[-suffix_len:], cur_tag) not in self.feature_rep_dict["f101_6"]:
                            self.feature_rep_dict["f101_6"][(cur_word[-suffix_len:], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f101_6"][(cur_word[-suffix_len:], cur_tag)] += 1
                    
                    for prefix_len in range(1, word_cutting_bound_6):
                        if (cur_word[:prefix_len], cur_tag) not in self.feature_rep_dict["f102_6"]:
                            self.feature_rep_dict["f102_6"][(cur_word[:prefix_len], cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f102_6"][(cur_word[:prefix_len], cur_tag)] += 1

                    # is numeric
                    is_all_numeric, is_combined_numeric = search_numbers_in_word(cur_word)
                    if is_all_numeric or is_combined_numeric:
                        if is_all_numeric:
                            numeric_feature = "f_is_numeric"
                        else:
                            numeric_feature = "f_is_combined_numeric"
                        if cur_tag not in self.feature_rep_dict[numeric_feature]:
                            self.feature_rep_dict[numeric_feature][cur_tag] = 1
                        else:
                            self.feature_rep_dict[numeric_feature][cur_tag] += 1
                    
                    # capital letters
                    is_all_capital, first_capital = check_capital(cur_word)
                    if is_all_capital or first_capital:
                        if is_all_capital:
                            capital_feature = "f_all_capital"
                        else:
                            capital_feature = "f_first_capital"
                        if cur_tag not in self.feature_rep_dict[capital_feature]:
                            self.feature_rep_dict[capital_feature][cur_tag] = 1
                        else:
                            self.feature_rep_dict[capital_feature][cur_tag] += 1

                    # length
                    if (cur_word_len, cur_tag) not in self.feature_rep_dict["f_length"]:
                        self.feature_rep_dict["f_length"][(cur_word_len, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f_length"][(cur_word_len, cur_tag)] += 1

                    # has_hyphen
                    if "-" in cur_word: 
                        if cur_tag not in self.feature_rep_dict["f_has_hyphen"]:
                            self.feature_rep_dict["f_has_hyphen"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["f_has_hyphen"][cur_tag] += 1
                    
                    # has_dot
                    if "." in cur_word and search_numbers_in_word(cur_word) == (False, False): 
                        if cur_tag not in self.feature_rep_dict["f_has_dot"]:
                            self.feature_rep_dict["f_has_dot"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["f_has_dot"][cur_tag] += 1
                    
                    # has_apostrophe
                    if "'" in cur_word: 
                        if cur_tag not in self.feature_rep_dict["f_has_apostrophe"]:
                            self.feature_rep_dict["f_has_apostrophe"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["f_has_apostrophe"][cur_tag] += 1

                    if adjective_suffixes(cur_word):
                        if cur_tag not in self.feature_rep_dict["f_has_adj_suffix"]:
                            self.feature_rep_dict["f_has_adj_suffix"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["f_has_adj_suffix"][cur_tag] += 1
                    
                    if plural_suffixes(cur_word):
                        if cur_tag not in self.feature_rep_dict["f_has_plural_suffix"]:
                            self.feature_rep_dict["f_has_plural_suffix"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["f_has_plural_suffix"][cur_tag] += 1

                    if past_suffixes(cur_word):
                        if cur_tag not in self.feature_rep_dict["f_has_past_suffix"]:
                            self.feature_rep_dict["f_has_past_suffix"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["f_has_past_suffix"][cur_tag] += 1

                    if present_suffix(cur_word):
                        if cur_tag not in self.feature_rep_dict["f_has_present_suffix"]:
                            self.feature_rep_dict["f_has_present_suffix"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["f_has_present_suffix"][cur_tag] += 1

                    if medical_suffix(cur_word):
                        if cur_tag not in self.feature_rep_dict["f_has_medical_suffix"]:
                            self.feature_rep_dict["f_has_medical_suffix"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["f_has_medical_suffix"][cur_tag] += 1

                    # mid capital
                    if check_mid_word_capital(cur_word):
                        if cur_tag not in self.feature_rep_dict["f_mid_capital"]:
                            self.feature_rep_dict["f_mid_capital"][cur_tag] = 1
                        else:
                            self.feature_rep_dict["f_mid_capital"][cur_tag] += 1

                    if is_preposition(cur_word_lower):
                        if (cur_word_lower, cur_tag) not in self.feature_rep_dict["f_is_preposition"]:
                            self.feature_rep_dict["f_is_preposition"][(cur_word_lower, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f_is_preposition"][(cur_word_lower, cur_tag)] += 1

                # w[-2] = w[-1] = "*"
                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                # end of sentence: w[n+1] = "~"
                sentence.append(("~", "~"))

                for i in range(2, len(sentence) - 1):
                    # history = (w[current], t[current], w[-1], t[-1], w[-2], t[-2], w[+1])
                    cur_word, cur_tag = sentence[i][WORD], sentence[i][TAG]
                    cur_word_lower = cur_word.lower()
                    prev_word, prev_tag = sentence[i - 1][WORD], sentence[i - 1][TAG]
                    prev_word_lower = prev_word.lower()
                    prev_prev_word, prev_prev_tag = sentence[i - 2][WORD], sentence[i - 2][TAG]
                    next_word = sentence[i + 1][WORD]
                    next_word_len = len(next_word)

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
                    
                    if (prev_word, prev_tag, cur_tag) not in self.feature_rep_dict["f106_prev_tag"]:
                        self.feature_rep_dict["f106_prev_tag"][(prev_word, prev_tag, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f106_prev_tag"][(prev_word, prev_tag, cur_tag)] += 1

                    if (next_word, cur_tag) not in self.feature_rep_dict["f107"]:
                        self.feature_rep_dict["f107"][(next_word, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f107"][(next_word, cur_tag)] += 1

                    # f_next_length
                    if (next_word_len, cur_tag) not in self.feature_rep_dict["f_next_length"]:
                        self.feature_rep_dict["f_next_length"][(next_word_len, cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f_next_length"][(next_word_len, cur_tag)] += 1
                    
                     # prepositions
                    if is_preposition(prev_word_lower):
                        if (prev_word_lower, cur_tag) not in self.feature_rep_dict["f_prev_preposition"]:
                            self.feature_rep_dict["f_prev_preposition"][(prev_word_lower, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f_prev_preposition"][(prev_word_lower, cur_tag)] += 1

                    # determiner
                    if is_determiner(prev_word_lower):
                        if (prev_word_lower, cur_tag) not in self.feature_rep_dict["f_prev_determiner"]:
                            self.feature_rep_dict["f_prev_determiner"][(prev_word_lower, cur_tag)] = 1
                        else:
                            self.feature_rep_dict["f_prev_determiner"][(prev_word_lower, cur_tag)] += 1

                    self.histories.append(history)


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int, filtered_feature_list: List[str]):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.feature_to_idx = {feature: OrderedDict() for feature in filtered_feature_list}
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
    c_word_len = len(c_word)
    c_word_lower = c_word.lower()
    c_tag = history[1]
    prev_word = history[2]
    prev_word_lower = prev_word.lower()
    prev_tag = history[3]
    prev_prev_tag = history[5]
    next_word = history[6]
    next_word_len = len(next_word)
    features = []

    # f100
    if "f100" in dict_of_dicts:
        if (c_word, c_tag) in dict_of_dicts["f100"]:
            features.append(dict_of_dicts["f100"][(c_word, c_tag)])

    # calculating maximum cut size for prefix and suffix
    # f101
    word_cutting_bound_4 = min(len(c_word) + 1, 5)
    word_cutting_bound_5 = min(len(c_word) + 1, 6)
    word_cutting_bound_6 = min(len(c_word) + 1, 7)
    # 101-102
    if "f101" in dict_of_dicts:
        for suffix_len in range(1, word_cutting_bound_4):
            word_suffix = c_word[-suffix_len:]
            if (word_suffix, c_tag) in dict_of_dicts["f101"]:
                features.append(dict_of_dicts["f101"][(word_suffix, c_tag)])

    if "f102" in dict_of_dicts:
        for prefix_len in range(1, word_cutting_bound_4):
            word_prefix = c_word[:prefix_len]
            if (word_prefix, c_tag) in dict_of_dicts["f102"]:
                features.append(dict_of_dicts["f102"][(word_prefix, c_tag)])

    # 101_5-102_5
    if "f101_5" in dict_of_dicts:
        for suffix_len in range(1, word_cutting_bound_5):
            word_suffix = c_word[-suffix_len:]
            if (word_suffix, c_tag) in dict_of_dicts["f101_5"]:
                features.append(dict_of_dicts["f101_5"][(word_suffix, c_tag)])

    if "f102_5" in dict_of_dicts:
        for prefix_len in range(1, word_cutting_bound_5):
            word_prefix = c_word[:prefix_len]
            if (word_prefix, c_tag) in dict_of_dicts["f102_5"]:
                features.append(dict_of_dicts["f102_5"][(word_prefix, c_tag)])

    # 101_6-102_6
    if "f101_6" in dict_of_dicts:
        for suffix_len in range(1, word_cutting_bound_6):
            word_suffix = c_word[-suffix_len:]
            if (word_suffix, c_tag) in dict_of_dicts["f101_6"]:
                features.append(dict_of_dicts["f101_6"][(word_suffix, c_tag)])

    if "f102_6" in dict_of_dicts:
        for prefix_len in range(1, word_cutting_bound_6):
            word_prefix = c_word[:prefix_len]
            if (word_prefix, c_tag) in dict_of_dicts["f102_6"]:
                features.append(dict_of_dicts["f102_6"][(word_prefix, c_tag)])
            
    # f103
    if "f103" in dict_of_dicts: 
        if (prev_prev_tag, prev_tag, c_tag) in dict_of_dicts["f103"]:
            features.append(dict_of_dicts["f103"][(prev_prev_tag, prev_tag,c_tag)])

    # f104
    if "f104" in dict_of_dicts: 
        if (prev_tag, c_tag) in dict_of_dicts["f104"]:
            features.append(dict_of_dicts["f104"][(prev_tag, c_tag)])

    # f105
    if "f105" in dict_of_dicts:
        if c_tag in dict_of_dicts["f105"]:
            features.append(dict_of_dicts["f105"][c_tag])

    # f106
    if "f106" in dict_of_dicts:
        if (prev_word, c_tag) in dict_of_dicts["f106"]:
            features.append(dict_of_dicts["f106"][(prev_word, c_tag)])
    
    if "f106_prev_tag" in dict_of_dicts:
        if (prev_word, prev_tag, c_tag) in dict_of_dicts["f106_prev_tag"]:
            features.append(dict_of_dicts["f106_prev_tag"][(prev_word, prev_tag, c_tag)])

    # f107
    if "f107" in dict_of_dicts:
        if (next_word, c_tag) in dict_of_dicts["f107"]:
            features.append(dict_of_dicts["f107"][(next_word, c_tag)])

    is_all_numeric, is_combined_numeric = search_numbers_in_word(c_word)
    is_all_capital, first_capital = check_capital(c_word)
    # is numeric
    if "f_is_numeric" in dict_of_dicts:
        if is_all_numeric and c_tag in dict_of_dicts["f_is_numeric"]:
            features.append(dict_of_dicts["f_is_numeric"][c_tag])
    
    # is combined numeric
    if "f_is_combined_numeric" in dict_of_dicts :
        if is_combined_numeric and c_tag in dict_of_dicts["f_is_combined_numeric"]:
            features.append(dict_of_dicts["f_is_combined_numeric"][c_tag])

    # is numeric
    if "f_all_capital" in dict_of_dicts:
        if is_all_capital and c_tag in dict_of_dicts["f_all_capital"]:
            features.append(dict_of_dicts["f_all_capital"][c_tag])
    
    # is combined numeric
    if "f_first_capital" in dict_of_dicts:
        if first_capital and c_tag in dict_of_dicts["f_first_capital"]:
            features.append(dict_of_dicts["f_first_capital"][c_tag])

     # length
    if "f_length" in dict_of_dicts:
        if (c_word_len, c_tag) in dict_of_dicts["f_length"]:
            features.append(dict_of_dicts["f_length"][(c_word_len, c_tag)])

    # next length
    if "f_next_length" in dict_of_dicts:
        if (next_word_len, c_tag) in dict_of_dicts["f_next_length"]:
            features.append(dict_of_dicts["f_next_length"][(next_word_len, c_tag)])

    # has hyphen
    if "f_has_hyphen" in dict_of_dicts:  
        if c_tag in dict_of_dicts["f_has_hyphen"] and "-" in c_word:
            features.append(dict_of_dicts["f_has_hyphen"][c_tag])

    # has dot
    if "f_has_dot" in dict_of_dicts:  
        if c_tag in dict_of_dicts["f_has_dot"] and "." in c_word and search_numbers_in_word(c_word) == (False, False):
            features.append(dict_of_dicts["f_has_dot"][c_tag])

    # has apostrophe
    if "f_has_apostrophe" in dict_of_dicts:  
        if c_tag in dict_of_dicts["f_has_apostrophe"] and "'" in c_word:
            features.append(dict_of_dicts["f_has_apostrophe"][c_tag])

    # mid capital
    if "f_mid_capital" in dict_of_dicts:
        if c_tag in dict_of_dicts["f_mid_capital"] and check_mid_word_capital(c_word):
            features.append(dict_of_dicts["f_mid_capital"][c_tag])

    # prepositions
    if "f_prev_preposition" in dict_of_dicts:
        if (prev_word_lower, c_tag) in dict_of_dicts["f_prev_preposition"]:
            features.append(dict_of_dicts["f_prev_preposition"][(prev_word_lower, c_tag)])
    
    # determiner
    if "f_prev_determiner" in dict_of_dicts:
        if (prev_word_lower, c_tag) in dict_of_dicts["f_prev_determiner"]:
            features.append(dict_of_dicts["f_prev_determiner"][(prev_word_lower, c_tag)])

    # special suffixes 
    if "f_has_adj_suffix" in dict_of_dicts:
        if c_tag in dict_of_dicts["f_has_adj_suffix"] and adjective_suffixes(c_word):
            features.append(dict_of_dicts["f_has_adj_suffix"][c_tag])

    if "f_has_plural_suffix" in dict_of_dicts:
        if c_tag in dict_of_dicts["f_has_plural_suffix"] and plural_suffixes(c_word):
            features.append(dict_of_dicts["f_has_plural_suffix"][c_tag])

    if "f_has_past_suffix" in dict_of_dicts:
        if c_tag in dict_of_dicts["f_has_past_suffix"] and past_suffixes(c_word):
            features.append(dict_of_dicts["f_has_past_suffix"][c_tag])
    
    if "f_has_present_suffix" in dict_of_dicts:
        if c_tag in dict_of_dicts["f_has_present_suffix"] and present_suffix(c_word):
            features.append(dict_of_dicts["f_has_present_suffix"][c_tag])

    if "f_has_medical_suffix" in dict_of_dicts:
        if c_tag in dict_of_dicts["f_has_medical_suffix"] and medical_suffix(c_word):
            features.append(dict_of_dicts["f_has_medical_suffix"][c_tag])

    # is preposition
    if "f_is_preposition" in dict_of_dicts:
        if (c_word_lower, c_tag) in dict_of_dicts["f_is_preposition"] and is_preposition(c_word_lower):
            features.append(dict_of_dicts["f_is_preposition"][(c_word_lower, c_tag)])

    return features


def preprocess_train(train_path, threshold, run_mode="test1"):
    # list of features for each run mode
    if run_mode == "test1":
        filtered_feature_list = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107", "f_is_numeric", "f_is_combined_numeric", "f_all_capital", "f_first_capital",
                              "f_length", "f_has_hyphen",  "f_has_dot", "f_has_apostrophe", "f_has_adj_suffix", "f_has_plural_suffix", "f_prev_determiner", "f_next_length",
                              "f_prev_preposition", "f_is_preposition"] 
    elif run_mode == 'train1':
        filtered_feature_list = ["f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107", "f_is_numeric", "f_is_combined_numeric", "f_all_capital", "f_first_capital",
                              "f_length", "f_has_hyphen",  "f_has_dot", "f_has_apostrophe", "f_has_adj_suffix", "f_has_plural_suffix", "f_prev_determiner", "f_next_length",
                              "f_prev_preposition", "f_is_preposition"] 
    elif run_mode == "comp1":
        filtered_feature_list = ["f100", "f101_5", "f102_5", "f103", "f104", "f105", "f106_prev_tag", "f107", "f_is_numeric", "f_is_combined_numeric", "f_all_capital", "f_first_capital",
                              "f_length", "f_has_hyphen",  "f_has_dot", "f_has_apostrophe", "f_has_adj_suffix", "f_has_plural_suffix", "f_prev_determiner", "f_next_length",
                              "f_prev_preposition", "f_is_preposition"] 
    elif run_mode == "train2":
        filtered_feature_list = ["f100", "f101_6", "f102_6", "f103", "f104", "f105", "f106", "f107",  "f_is_numeric", "f_is_combined_numeric",
                                "f_all_capital", "f_first_capital", "f_mid_capital", "f_has_hyphen",  "f_has_dot", "f_has_apostrophe", "f_prev_determiner", "f_prev_preposition", "f_is_preposition",
                                "f_has_adj_suffix", "f_has_past_suffix", "f_has_present_suffix", "f_has_medical_suffix"]
    elif run_mode == "comp2":
        filtered_feature_list = ["f100", "f101_6", "f102_6", "f103", "f104", "f105", "f106", "f107",  "f_is_numeric", "f_is_combined_numeric",
                                "f_all_capital", "f_first_capital", "f_mid_capital", "f_has_hyphen",  "f_has_dot", "f_has_apostrophe", "f_prev_determiner", "f_prev_preposition", "f_is_preposition",
                                "f_has_adj_suffix", "f_has_past_suffix", "f_has_present_suffix", "f_has_medical_suffix"]
    else:
        raise ValueError(d=f"Unknown run_mode. Expected one of [train1, train2, test1, comp1, comp2], got {run_mode}")
    
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    # feature2id
    feature2id = Feature2id(statistics, threshold, filtered_feature_list)
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
