import pickle
from inference import tag_all_test
from preprocessing import preprocess_train
from optimization import get_optimal_vector


def train_and_tag_comp(train_path: str, comp_path: str, threshold: int, lam: float, b: int) -> None:
    """Train models and tag the competition files 

    Args:
        train_path (str): path to train file
        comp_path (str): path to competition file
        threshold (int): features frequency threshold
        lam (float): regularization hyperparameter
        b (int): beam size for viterbi algorithm
    """
    if "1" in comp_path:
        run_mode = "comp1"
        predictions_path = 'comp_m1_206713612_316111442.wtag'
    else:
        run_mode = "comp2"
        predictions_path = 'comp_m2_206713612_316111442.wtag'
    weights_path=f'weights_{run_mode}.pkl'
    statistics, feature2id = preprocess_train(train_path, threshold, run_mode=run_mode)
    # get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]
    tag_all_test(comp_path, pre_trained_weights, feature2id, predictions_path, b=b)


def concatenate_files(input_files: list, output_file: str) -> None:
    """Concatenate 2 files content into one (used for concating train1 and test1 for training a model for comp1 inference)

    Args:
        input_files (list): list of files to concatenate
        output_file (str): path to the result file
    """
    with open(output_file, 'w') as outfile:
        for input_file in input_files:
            with open(input_file, 'r') as infile:
                outfile.write(infile.read())


if __name__ == "__main__":
    # concatenate_files(input_files=['data/train1.wtag', 'data/test1.wtag'], output_file='data/combined_model1_file.wtag')
    train_and_tag_comp(train_path='data/combined_model1_file.wtag', comp_path="data/comp1.words",threshold=1, lam=0.7, b=100)
    train_and_tag_comp(train_path='data/train2.wtag', comp_path="data/comp2.words",threshold=1, lam=0.05, b=100)
