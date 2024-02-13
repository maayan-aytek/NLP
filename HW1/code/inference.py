from tqdm import tqdm
from typing import List, Dict, Tuple, Union
import numpy as np
from preprocessing import Feature2id, represent_input_with_features, read_test
from collections import defaultdict


def update_q_dict(q_dict: dict, sentence: List[str], k:int, pre_trained_weights: np.ndarray, feature2id: Feature2id, all_tags: list, prev_tag:str, prev_prev_tag:str) -> None:
    """q values calculations

    Args:
        q_dict (dict): q dictionary before update
        sentence (List[str]): current sentence
        k (int): word index
        pre_trained_weights (np.ndarray): features weights vector
        feature2id (Feature2id): Feature2id object
        all_tags (list): list of all tags
        prev_tag (str): tag of previous word
        prev_prev_tag (str):  tag of previous previous word
    """
    all_exp_sum = 0
    curr_word = sentence[k]
    prev_word = sentence[k - 1]
    prev_prev_word = sentence[k - 2]
    next_word = sentence[k + 1] if k != len(sentence) - 1 else "~"
    for tag in all_tags: 
        history = (curr_word, tag, prev_word, prev_tag, prev_prev_word, prev_prev_tag, next_word)
        features = represent_input_with_features(history, feature2id.feature_to_idx) # extracting relevant features vector
        curr_exp = np.exp(pre_trained_weights[features].sum()) # numerator
        all_exp_sum += curr_exp # summing all numerators
        q_dict[(prev_prev_tag, prev_tag, tag, k)] = curr_exp
    
    for tag in all_tags: 
        q_dict[(prev_prev_tag, prev_tag, tag, k)] /= all_exp_sum # denominator devision


def viterbi_initialization(n: int, all_tags: List[str]) -> Tuple[dict, dict, List[str]]:
    """Initialization of pi and backpointers dictionaries in viterbi algorithm

    Args:
        n (int): sentence length
        all_tags (List[str]): list of all tags

    Returns:
        Tuple[dict, dict, List[str]]: initialized pi dict, bp dict and preds list
    """
    pi_dict = {}
    bp_dict = {}
    for k in range(n):
        for prev_tag in all_tags:
            for prev_prev_tag in all_tags:
                pi_dict[(k, prev_prev_tag, prev_tag)] = 0
                bp_dict[(k, prev_prev_tag, prev_tag)] = ""
    preds = ["~"] * n 
    return pi_dict, bp_dict, preds


def memm_viterbi(sentence: List[str], pre_trained_weights: np.ndarray, feature2id: Feature2id) -> List[str]:
    """Implementing MEMM viterbi algorithm

    Args:
        sentence (List[str]): list of words in the sentence
        pre_trained_weights (np.ndarray): pre trained weights vector
        feature2id (Feature2id): Feature2id object

    Returns:
        List[str]: predictions list
    """
    sentence = sentence[:-1] # remove "~" from sentence
    n = len(sentence)
    all_tags = list(feature2id.feature_statistics.tags)
    pi_dict, bp_dict, q_dict, preds, n_pi_values = first_iter_viterbi(sentence, n, all_tags, pre_trained_weights, feature2id)
    
    for k in range(4, n):
        for prev_prev_tag in all_tags:
            for prev_tag in all_tags:
                update_q_dict(q_dict, sentence, k, pre_trained_weights, feature2id, all_tags, prev_tag, prev_prev_tag)
            
        for prev_tag in all_tags:
            for curr_tag in all_tags: 
                values = []
                for prev_prev_tag in all_tags:
                    curr_value = q_dict[(prev_prev_tag, prev_tag, curr_tag, k)] * pi_dict[(k-1, prev_prev_tag, prev_tag)]
                    values.append([prev_prev_tag, curr_value])
                bp_dict[(k, prev_tag, curr_tag)], pi_dict[(k, prev_tag, curr_tag)] = max(values, key=lambda x: x[1])
                if k == n-1:
                    n_pi_values.append([prev_tag, curr_tag, pi_dict[(k, prev_tag, curr_tag)]])
    # setting last 2 words tags 
    preds[n-2], preds[n-1], _ = max(n_pi_values, key=lambda x: x[2])
    # setting rest of sentence tags backwards 
    for k in range(n-3, -1, -1):
        preds[k] = bp_dict[(k + 2, preds[k + 1], preds[k + 2])]
    # removing "*" tag
    return preds[2:]


def first_iter_viterbi(sentence: List[str], n: int, all_tags: List[str], pre_trained_weights: np.ndarray, feature2id: Feature2id, 
                       b: int = None) -> Union[Tuple[dict, dict, dict, dict, List[str], List], Tuple[dict, dict, dict, List[str]]]:
    """Initializa viterbi algorithm and run first step(s).

    Args:
        sentence (List[str]): list of sentence words 
        n (int): sentence length
        all_tags (List[str]): list of all tags 
        pre_trained_weights (np.ndarray): features weights vector
        feature2id (Feature2id): Feature2id object
        b (int, optional): beam size (in case running with beam search)
    Returns:
        Union[Tuple[dict, dict, dict, dict, List[str]], Tuple[dict, dict, dict, List[str]]]: 
        - pi_dict: probabilities dictionary 
        - bp_dict: back pointers (to prev prev tag) dictionary
        - q_dict: q values dictionary
        - best_tags_pairs (optional): dictionary of the best b states that will continue to the next iteration
        - preds: list of initialized predictions
        - n_pi_values: list of last word probabilites
    """
    pi_dict, bp_dict, preds = viterbi_initialization(n, all_tags)
    q_dict = {}
    n_pi_values = []
    # Previous tags initializations 
    prev_prev_tag = "*"
    prev_tag = "*"
    # k = 2
    update_q_dict(q_dict, sentence, 2, pre_trained_weights, feature2id, all_tags, prev_tag, prev_prev_tag)
    values = []
    for curr_tag in all_tags:
        curr_value = q_dict[(prev_prev_tag, prev_tag, curr_tag, 2)] * 1
        values.append([prev_tag, curr_tag, curr_value])
        bp_dict[(2, prev_tag, curr_tag)], pi_dict[(2, prev_tag, curr_tag)] = prev_prev_tag, curr_value
        if n == 3:
            n_pi_values.append([prev_tag, curr_tag, pi_dict[(2, prev_tag, curr_tag)]])
    if b: # beam search- find options for the next iteration
        sorted_values = sorted(values, key= lambda x: x[2], reverse=True)[:b]
        best_tags_pairs = defaultdict(list)
        for prev_tag, curr_tag, _ in sorted_values:
            best_tags_pairs[curr_tag].append(prev_tag)
        return pi_dict, bp_dict, q_dict, best_tags_pairs, preds, n_pi_values
    elif n > 3: # continue to k = 3 if needed
        for prev_tag in all_tags:
            update_q_dict(q_dict, sentence, 3, pre_trained_weights, feature2id, all_tags, prev_tag, prev_prev_tag)
        for prev_tag in all_tags:
            for curr_tag in all_tags:
                curr_value = q_dict[(prev_prev_tag, prev_tag, curr_tag, 3)] * pi_dict[(2, prev_prev_tag, prev_tag)]
                bp_dict[(3, prev_tag, curr_tag)], pi_dict[(3, prev_tag, curr_tag)] = prev_prev_tag, curr_value
                if n == 4:
                    n_pi_values.append([prev_tag, curr_tag, pi_dict[(3, prev_tag, curr_tag)]])

    return pi_dict, bp_dict, q_dict, preds, n_pi_values


def memm_beam_search(sentence: List[str], pre_trained_weights: np.ndarray, feature2id: Feature2id, b: int) -> List[str]:
    """Implementing MEMM viterbi algorithm with beam search

    Args:
        sentence (List[str]): list of words in the sentence
        pre_trained_weights (np.ndarray): pre trained weights vector
        feature2id (Feature2id): Feature2id object
        b (int): beam size

    Returns:
        List[str]: predictions list
    """
    sentence = sentence[:-1] # remove "~" from sentence
    n = len(sentence)
    all_tags = list(feature2id.feature_statistics.tags)
    pi_dict, bp_dict, q_dict, best_tags_pairs, preds, n_pi_values = first_iter_viterbi(sentence, n, all_tags, pre_trained_weights, feature2id, b)
   
    for k in range(3, n):
        for prev_tag, prev_prev_tags_list in best_tags_pairs.items():
                for prev_prev_tag in prev_prev_tags_list:
                    update_q_dict(q_dict, sentence, k, pre_trained_weights, feature2id, all_tags, prev_tag, prev_prev_tag)
        
        values = [] 
        for curr_tag in all_tags:
            for prev_tag, prev_prev_tags_list in best_tags_pairs.items():
                prev_values = []
                for prev_prev_tag in prev_prev_tags_list:
                    curr_value = q_dict[(prev_prev_tag, prev_tag, curr_tag, k)] * pi_dict[(k-1, prev_prev_tag, prev_tag)]
                    values.append([prev_tag, curr_tag, curr_value])
                    prev_values.append([prev_prev_tag, curr_value])
                
                bp_dict[(k, prev_tag, curr_tag)], pi_dict[(k, prev_tag, curr_tag)] = max(prev_values, key=lambda x: x[1])

                if k == n-1:
                    n_pi_values.append([prev_tag, curr_tag, pi_dict[(k, prev_tag, curr_tag)]])

        sorted_values = sorted(values, key= lambda x: x[2], reverse=True)[:b]
        best_tags_pairs = defaultdict(list)
        for prev_tag, curr_tag, _ in sorted_values:
            best_tags_pairs[curr_tag].append(prev_tag)
        

    # setting last 2 words tags 
    preds[n-2], preds[n-1], _ = max(n_pi_values, key=lambda x: x[2])
    # setting rest of sentence tags backwards 
    for k in range(n-3, -1, -1):
        preds[k] = bp_dict[(k + 2, preds[k + 1], preds[k + 2])]
    # removing "*" tag
    return preds[2:]


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path, b=5):
    tagged = "test" in test_path or "train" in test_path
    test = read_test(test_path, tagged=tagged)
    all_tags = list(feature2id.feature_statistics.tags)
    output_file = open(predictions_path, "a+")
    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_beam_search(sentence, pre_trained_weights, feature2id, b=b)
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()





    
    
    
        

