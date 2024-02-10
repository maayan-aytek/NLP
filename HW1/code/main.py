import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test
from utils import eval_preds, k_fold_cv


def main():
    threshold = 1 
    run_mode = "test1"
    lam = 1 if "1" in run_mode else 0.01
    train_path = "data/train1.wtag" if "1" in run_mode else "data/train2.wtag"
    if run_mode == "test1":
        test_path = "data/test1.wtag" 
    elif run_mode == "comp1":
        test_path = "data/comp1.wtag" 
    elif run_mode == "comp2":
        test_path = "data/comp2.wtag" 
    else:
        raise ValueError("unkown run mode")

    print("RUN MODE:", run_mode)
    print("TRAIN PATH:", train_path)
    weights_path = f'weights_{run_mode}.pkl' 
    predictions_path = f'predictions_{run_mode}.wtag'

    statistics, feature2id = preprocess_train(train_path, threshold, run_mode=run_mode)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    print(pre_trained_weights)
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)
    accuracy_score, top_10_mistakes_dict = eval_preds(test_path, predictions_path)
    print("Accuracy:", accuracy_score*100)
    print("Top 10 mistakes:", top_10_mistakes_dict)

    # k_fold_cv("data/train2.wtag", weights_path, k_folds=4)


if __name__ == '__main__':
    main()
