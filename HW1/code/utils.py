from typing import Dict, Tuple
import numpy as np
from preprocessing import read_test, preprocess_train
from inference import tag_all_test
from optimization import get_optimal_vector
import pickle


TAG = 1
WORD = 0


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
    word_mistakes_matrix = {}
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
                if test_label not in confusion_matrix.keys():
                    confusion_matrix[test_label] = [pred_label]
                else:
                    confusion_matrix[test_label].append(pred_label)
                if (test_label, pred_label) not in word_mistakes_matrix.keys():
                    word_mistakes_matrix[(test_label, pred_label)] = [test[WORD][i + 2]]
                else:
                    word_mistakes_matrix[(test_label, pred_label)].append(test[WORD][i + 2])
    accuracy_score = count_true_preds / count_all_preds
    return accuracy_score, confusion_matrix, word_mistakes_matrix



def display_top_10_mistakes(confusion_matrix: Dict) -> None:
    """Display confusion matrix of top 10 classes with mistakes

    Args:
        confusion_matrix (Dict): _description_
    """
    def gradual_color(val):
        if val == 0:
            color = '#00FF00'
        elif val < 15:
            color = 'yellow'
        elif 15 <= val <= 30:
            color = 'orange'
        else:
            color = '#FF5733'
        return f'color: {color}'

    import pandas as pd
    mistake_counts = {}
    for true_label, mistakes in confusion_matrix.items():
        if true_label not in mistake_counts:
            mistake_counts[true_label] = {}
        for mistake_label in mistakes:
            if mistake_label not in mistake_counts[true_label]:
                mistake_counts[true_label][mistake_label] = 1
            else:
                mistake_counts[true_label][mistake_label] += 1

    sorted_classes = sorted(mistake_counts.items(), key=lambda x: sum(x[1].values()), reverse=True)[:10]

    all_labels = set(dict(sorted_classes).keys())
    for true_label, mistakes in sorted_classes:
        all_labels.update(set(mistakes.keys()))

    df = pd.DataFrame(index=[x[0] for x in sorted_classes], columns=all_labels)

    for true_label, mistakes in sorted_classes:
        for mistake_label, count in mistakes.items():
            df.at[true_label, mistake_label] = count
            
    df.fillna(0, inplace=True)

    sorted_columns = [col for col in df.index if col in df.columns]
    sorted_columns += [col for col in df.columns if col not in sorted_columns]
    df = df.reindex(columns=sorted_columns)
    styled_df = df.style.applymap(gradual_color)
    print(styled_df)



def k_fold_cv(data_path: str, weights_path:str, k_folds: int, run_mode: str, threshold: int=1, lam: int=1) -> None:
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
    sentences = [sen[WORD][2:-1] for sen in file]
    indices = np.array([i for i in range(len(sentences))]).astype(int) 
    np.random.seed(100)
    np.random.shuffle(indices)  
    fold_size = len(sentences) // k_folds
    accuracy_list = [] 
    for i in range(k_folds):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        train_data = [sentences[idx] for idx in train_indices]
        test_data = [sentences[idx] for idx in test_indices]

        # build fold train data
        output_file = open(f"fold_{i}_train.wtag", "w")
        for sentence in train_data:
            output_file.write(" ".join(sentence) + "\n")
        output_file.close()

        # build fold test data
        output_file = open(f"fold_{i}_test.wtag", "w")
        for sentence in test_data:
            output_file.write(" ".join(sentence) + "\n")
        output_file.close()

        # train the model
        statistics, feature2id = preprocess_train(f"fold_{i}_train.wtag", threshold, run_mode=run_mode)
        get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

        with open(weights_path, 'rb') as f:
            optimal_params, feature2id = pickle.load(f)
        pre_trained_weights = optimal_params[0]

        # predict and eval
        tag_all_test(f"fold_{i}_test.wtag", pre_trained_weights, feature2id, f"predictions_fold_{i}.wtag")
        accuracy, _, _, = eval_preds(f"fold_{i}_test.wtag", f"predictions_fold_{i}.wtag")
        accuracy_list.append(accuracy)

        print(f"Fold {i} Accuracy: {accuracy_list[i]*100}")

    for i in range(k_folds):
        print(f"Fold {i} Accuracy: {accuracy_list[i]*100}")
    print(f"Average Accuracy: {(sum(accuracy_list)/len(accuracy_list)) * 100}")