import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test
from utils import eval_preds, k_fold_cv
import time
from generate_comp_tagged import train_and_tag_comp


def main():
    # concatenate_files(input_files=['data/train1.wtag', 'data/test1.wtag'], output_file='data/combined_model1_file.wtag')
    train_and_tag_comp(train_path='data/combined_model1_file.wtag', comp_path="data/comp1.words",threshold=1, lam=0.7, b=100)
    train_and_tag_comp(train_path='data/train2.wtag', comp_path="data/comp2.words",threshold=1, lam=0.05, b=100)

    # ------ This part is not relevant for the submission, we used it for our models evaluation -------
    # threshold = 1
    # run_mode = "train2"
    # lam = 0.7 if "1" in run_mode else 0.05
    # train_path = "data/train1.wtag" if "1" in run_mode else "data/train2.wtag"
    # if run_mode == "test1":
    #     test_path = "data/test1.wtag"
    # elif run_mode == "train1":
    #     test_path = "data/train1.wtag"
    # elif run_mode == "comp1":
    #     test_path = "data/comp1.wtag" 
    # elif run_mode == "train2":
    #     test_path = "data/train2.wtag" 
    # elif run_mode == "comp2":
    #     test_path = "data/comp2.wtag" 
    # else:
    #     raise ValueError("unkown run mode")

    # print("RUN MODE:", run_mode)
    # print("TRAIN PATH:", train_path)
    # weights_path = f'weights_{run_mode}.pkl' 
    # predictions_path = f'predictions_{run_mode}.wtag'

    # t0 = time.time()
    # statistics, feature2id = preprocess_train(train_path, threshold, run_mode=run_mode)
    # get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)
    # print(f"Training Time of {train_path}: {time.time() - t0}")

    # with open(weights_path, 'rb') as f:
    #     optimal_params, feature2id = pickle.load(f)
    # pre_trained_weights = optimal_params[0]

    # print(pre_trained_weights)
    # tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path)
    # accuracy_score, confusion_matrix, _ = eval_preds(test_path, predictions_path)
    # print("Accuracy:", accuracy_score*100)

    # k_fold_cv("data/train2.wtag", weights_path, k_folds=4, lam=lam, threshold=threshold, run_mode=run_mode)


if __name__ == '__main__':
    main()
