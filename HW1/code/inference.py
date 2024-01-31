from preprocessing import read_test
from tqdm import tqdm
from typing import List, Dict, Tuple 
import numpy as np
from preprocessing import Feature2id, represent_input_with_features
from collections import OrderedDict, defaultdict


def update_q_dict(q_dict: dict, sentence: List[str], k:int, pre_trained_weights: np.ndarray, feature2id: Feature2id, all_tags: list, prev_tag:str, prev_prev_tag:str) -> None:
    all_exp_sum = 0
    for tag in all_tags: #TODO: handle "*" and "~"
        history = (sentence[k], tag, sentence[k - 1], prev_tag, sentence[k - 2], prev_prev_tag, sentence[k + 1])
        if history not in feature2id.histories_features:
            features = represent_input_with_features(history, feature2id.feature_to_idx)
        else:
            features = feature2id.histories_features[history]
        curr_exp = np.exp(sum([pre_trained_weights[feature] for feature in features]))
        all_exp_sum += curr_exp
        q_dict[(prev_prev_tag, prev_tag, tag)] = curr_exp
    
    for tag in all_tags: 
        q_dict[(prev_prev_tag, prev_tag, tag)] /= all_exp_sum


def viterbi_initialization(n: int, all_tags: List[str]) -> Tuple[dict, dict, List[str]]:
    pi_dict = {}
    bp_dict = {}
    for k in range(n):
        for prev_tag in all_tags:
            for prev_prev_tag in all_tags:
                pi_dict[(k, prev_prev_tag, prev_tag)] = 0
                bp_dict[(k, prev_prev_tag, prev_tag)] = "error"
    pi_dict[(1, "*", "*")] = 1
    preds = ["~"] * n 
    return pi_dict, bp_dict, preds


def memm_viterbi(sentence: List[str], pre_trained_weights: np.ndarray, feature2id: Feature2id) -> List[str]:
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """
    n = len(sentence)
    all_tags = list(feature2id.feature_statistics.tags) + ["*"] #TODO: check!
    pi_dict, bp_dict, preds = viterbi_initialization(n, all_tags)
    for k in range(2, n-1):
        q_dict = {}
        for prev_prev_tag in all_tags:
            for prev_tag in all_tags:
                update_q_dict(q_dict, sentence, k, pre_trained_weights, feature2id, all_tags, prev_tag, prev_prev_tag)
            
        for prev_tag in all_tags:
            for curr_tag in all_tags: 
                values = []
                for prev_prev_tag in all_tags:
                    curr_value = q_dict[(prev_prev_tag, prev_tag, curr_tag)] * pi_dict[(k-1, prev_prev_tag, prev_tag)]
                    values.append([prev_prev_tag, curr_value])
                bp_dict[(k, prev_tag, curr_tag)], pi_dict[(k, prev_tag, curr_tag)] = max(values, key=lambda x: x[1])
    n_pi_values = []
    for prev_tag in all_tags:
        for curr_tag in all_tags:
            n_pi_values.append([prev_tag, curr_tag, pi_dict[(n-1, prev_tag, curr_tag)]])
    preds[n-1], preds[n-2], _ = max(n_pi_values, key=lambda x: x[2])
    for k in range(n-3, -1, -1): #TODO: handle 0 and 28 cases
        preds[k] = bp_dict[(k + 2, preds[k + 1], preds[k + 2])]
    return preds


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "a+")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
