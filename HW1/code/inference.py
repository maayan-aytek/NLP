from preprocessing import read_test
from tqdm import tqdm
from typing import List, Dict, Tuple, Union
import numpy as np
from preprocessing import Feature2id, represent_input_with_features
from collections import OrderedDict, defaultdict
from omri import memm_viterbi_omri, memm_viterbi1 #TODO: delete


TAG = 1

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
        if tag == "*": # prbability is zero for "*" tagging in the sentence itself (from the 3rd word and forth)
            q_dict[(prev_prev_tag, prev_tag, tag, k)] = 0
            continue
        history = (curr_word, tag, prev_word, prev_tag, prev_prev_word, prev_prev_tag, next_word)
            # extracting relevant features vector
        features = represent_input_with_features(history, feature2id.feature_to_idx)
        curr_exp = np.exp(pre_trained_weights[features].sum()) # numerator
        all_exp_sum += curr_exp # summing all numerators
        q_dict[(prev_prev_tag, prev_tag, tag, k)] = curr_exp
    
    for tag in all_tags: 
        # denominator devision
        q_dict[(prev_prev_tag, prev_tag, tag, k)] /= all_exp_sum


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
    pi_dict[(1, "*", "*")] = 1 # first step in the viterbi algorithm
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
    all_tags = list(feature2id.feature_statistics.tags) + ["*"] # adding "*" to the tags set
    pi_dict, bp_dict, preds = viterbi_initialization(n, all_tags)
    # updating pi and bp values
    n_pi_values = []
    q_dict = {}
    
    for k in range(2, n):
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
    all_tags = list(feature2id.feature_statistics.tags) + ["*"] # adding "*" to the tags set
    pi_dict, bp_dict, preds = viterbi_initialization(n, all_tags)
    # updating pi and bp values
    n_pi_values = []
    best_tags_pairs = {}
    for prev_tag in all_tags:
        best_tags_pairs[prev_tag] = all_tags
    q_dict = {}
    for k in range(2, n):
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

        sorted_values = sorted(values, key= lambda x: x[2], reverse=True)[:b]
        best_tags_pairs = defaultdict(list)
        for prev_tag, curr_tag, _ in sorted_values:
            best_tags_pairs[curr_tag].append(prev_tag)

        if k == n-1:
            n_pi_values.append([prev_tag, curr_tag, pi_dict[(k, prev_tag, curr_tag)]])

    # setting last 2 words tags 
    preds[n-2], preds[n-1], _ = max(n_pi_values, key=lambda x: x[2])
    # setting rest of sentence tags backwards 
    for k in range(n-3, -1, -1):
        preds[k] = bp_dict[(k + 2, preds[k + 1], preds[k + 2])]
    # removing "*" tag
    return preds[2:]


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")
    import time
    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        t0 = time.time()
        pred = memm_beam_search(sentence, pre_trained_weights, feature2id, b=5) 
        # memm_viterbi1(sentence, pre_trained_weights, feature2id, top_k=-1000)  # memm_viterbi(sentence, pre_trained_weights, feature2id)
         # memm_beam_search(sentence, pre_trained_weights, feature2id, b=5)# memm_viterbi_omri(sentence, pre_trained_weights, feature2id, top_k=5)
        print(time.time() - t0)
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()


def eval_preds(labeled_test_path: str, prediction_path: str) -> Tuple[float, Dict[Tuple[str, str], int]]:
    """Calculate total accuracy and top 10 mistakes confusion matrix

    Args:
        labeled_test_path (str): path to the test file
        prediction_path (str): path to the prediction file

    Returns:
        Tuple[float, Dict[Tuple[str, str]]: accuracy score, confusion matrix dict (key=pairs of tags, value=count of mistakes)
    """
    test_list = read_test(labeled_test_path, tagged=True)
    preds_list = read_test(prediction_path, tagged=True)
    confusion_matrix = {}
    count_true_preds = 0
    count_all_preds = 0
    for test, pred in zip(test_list, preds_list):
        test_labels = test[TAG][2:-1]
        pred_labels = pred[TAG][2:-1]
        assert len(test_labels) == len(pred_labels), "Test and Preds are not aligned"
        for i in range(len(test_labels)):
            test_label = test_labels[i]
            pred_label = pred_labels[i]
            count_all_preds += 1
            if test_label == pred_label:
                count_true_preds += 1
            else:
                if (test_label, pred_label) not in confusion_matrix.keys():
                    confusion_matrix[(test_label, pred_label)] = 1
                else:
                    confusion_matrix[(test_label, pred_label)] += 1
    sorted_mistakes = sorted(confusion_matrix.items(), key=lambda x: x[1], reverse=True)
    top_10_mistakes_dict = dict(sorted_mistakes[:10])
    accuracy_score = count_true_preds / count_all_preds
    return accuracy_score, top_10_mistakes_dict




    
    
    
        

