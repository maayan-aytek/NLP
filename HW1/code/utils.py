from typing import List, Dict, Tuple, Union
import numpy as np
from preprocessing import Feature2id, represent_input_with_features, read_test, preprocess_train
from inference import tag_all_test, memm_beam_search
from collections import OrderedDict, defaultdict
from optimization import get_optimal_vector
import pickle


TAG = 1

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


def k_fold_cv(data_path: str, weights_path:str, k_folds: int, threshold: int=1, lam: int=1) -> None:
    """
    Perform k-fold cross validation on a given labeled data.
    Args:
        data_path (str): path to the data
        weights_path (str): path to save the features weights
        k_folds (int): number of folds to split
        threshold (int, optional): threshold for features filtering. Defaults to 1.
        lam (int, optional): learning rate. Defaults to 1.
    """

    tagged = "test" in data_path
    file = read_test(data_path, tagged=tagged)
    sentences = [sen[TAG][2:-1] for sen in file]
    indices = [i for i in range(len(sentences))]  
    indices = np.random.shuffle(indices)  
    fold_size = len(sentences) // k_folds
    accuracy_list = [] 
    for i in range(k_folds):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        train_data = [sentences[idx] for idx in train_indices]
        test_data = [sentences[idx] for idx in test_indices]

        # build fold train data
        output_file = open("fold_train.wtag", "w")
        for sentence in train_data:
            output_file.write(" ".join(sentence) + "\n")
        output_file.close()

        # build fold test data
        output_file = open("fold_test.wtag", "w")
        for sentence in test_data:
            output_file.write(" ".join(sentence) + "\n")
        output_file.close()

        # train the model
        statistics, feature2id = preprocess_train("fold_train.wtag", threshold)
        get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

        with open(weights_path, 'rb') as f:
            optimal_params, feature2id = pickle.load(f)
        pre_trained_weights = optimal_params[0]

        # predict and eval
        tag_all_test("fold_test.wtag", pre_trained_weights, feature2id, f"predictions_fold_{i}.wtag")
        accuracy, _ = eval_preds("fold_test.wtag", f"predictions_fold_{i}.wtag")
        accuracy_list.append(accuracy)

        print(f"Fold {i} Accuracy: {accuracy_list[i]}")

    print(f"Average Accuracy: {sum(accuracy_list)/len(accuracy_list)}")