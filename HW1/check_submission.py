import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import zipfile
import shutil

COMP_FILES_PATH = 'comps files'
ID1 = input("insert ID1: ")
ID2 = input("insert ID2: ")

def unzip_directory(zip_path, output_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)

if not os.path.exists(COMP_FILES_PATH):
    os.makedirs(COMP_FILES_PATH)
    os.makedirs(f"{COMP_FILES_PATH}/{ID1}_{ID2}")

def compare_files(true_file, pred_file):
    with open(true_file, 'r') as f:
        true_data = [x.strip() for x in f.readlines() if x != '']
    with open(pred_file, 'r') as f:
        pred_data = [x.strip() for x in f.readlines() if x != '']
    if len(pred_data) != len(true_data):
        if len(pred_data) > len(true_data):
            pred_data = pred_data[:len(true_data)]
        else:
            raise KeyError
    num_correct, num_total = 0, 0
    prob_sent = set()
    predictions, true_labels = [], []
    for idx, sen in enumerate(true_data):
        pred_sen = pred_data[idx]
        if pred_sen.endswith('._.') and not pred_sen.endswith(' ._.'):
            pred_sen = pred_sen[:-3] + ' ._.'
        true_words = [x.split('_')[0] for x in sen.split()]
        true_tags = [x.split('_')[1] for x in sen.split()]
        true_labels += true_tags
        pred_words = [x.split('_')[0] for x in pred_sen.split()]
        try:
            pred_tags = [x.split('_')[1] for x in pred_sen.split()]
            predictions += pred_tags
        except IndexError:
            prob_sent.add(idx)
            pred_tags = []
            for x in pred_sen.split():
                if '_' in x:
                    pred_tags.append(x.split('_'))
                else:
                    pred_tags.append(None)
        if pred_words[-1] == '~':
            pred_words = pred_words[:-1]
            pred_tags = pred_tags[:-1]
        if pred_words != true_words:
            prob_sent.add(idx)
        elif len(pred_tags) != len(true_tags):
            prob_sent.add(idx)
        for i, (tt, tw) in enumerate(zip(true_tags, true_words)):
            num_total += 1
            if len(pred_words) > i:
                pw = pred_words[i]
                pt = pred_tags[i]
            else:
                prob_sent.add(idx)
                continue
            if pw != tw:
                continue
            if tt == pt:
                num_correct += 1
        pass
    labels = sorted(list(set(true_labels)))
    if len(prob_sent) > 0:
        print(prob_sent)
    conf_mat = confusion_matrix(y_true=true_labels, y_pred=predictions, labels=labels)
    conf_mat = pd.DataFrame(conf_mat, index=labels, columns=labels)
    return num_correct / num_total, prob_sent, conf_mat
    # return num_correct / num_total, prob_sent


def calc_scores(e):
    scores = []
    for sub in os.listdir(COMP_FILES_PATH):
        # print(sub)
        cur_dir = f'{COMP_FILES_PATH}/{sub}'
        comp1_file = [x for x in os.listdir(cur_dir) if x.startswith('comp_m1')]
        comp2_file = [x for x in os.listdir(cur_dir) if x.startswith('comp_m2')]
        prob1, prob2, ids = set(), set(), []
        if len(comp1_file) != 1:
            print(f'{sub} has a Problem with m1!')
            e = True
            comp1 = None
        else:
            ids = comp1_file[0].replace('comp_m1_', '').split('.')[0].split('_')
            comp1_file = f'{cur_dir}/{comp1_file[0]}'
            # comp1, prob1, conf_mat = compare_files('comp1.wtag', comp1_file)
            # conf_mat.to_csv('Comp 1 conf.csv')
            comp1, prob1, conf_mat1 = compare_files('data/comp1.wtag_public', comp1_file)
            print(conf_mat1)
            comp1 = round(comp1 * 100, 2)
        if len(comp2_file) != 1:
            print(f'{sub} ha a Problem with m2!')
            e = True
            comp2 = None
        else:
            ids = comp2_file[0].replace('comp_m2_', '').split('.')[0].split('_')
            comp2_file = f'{cur_dir}/{comp2_file[0]}'
            comp2, prob2, conf_mat2  = compare_files('data/comp2.wtag_public', comp2_file)
            # comp2, prob2 = compare_files('comp2.wtag', comp2_file)
            print(conf_mat2)
            # conf_mat.to_csv('Comp 2 conf.csv')
            comp2 = round(comp2 * 100, 2)

        cur_score = {f'ID {idx + 1}': cur_id for idx, cur_id in enumerate(ids)}
        cur_score['Comp 1 Acc'] = comp1
        cur_score['Comp 2 Acc'] = comp2
        print("Fake score for comp1: "+str(comp1))
        print("Fake score for comp2: "+str(comp2))
        if comp1 and comp1 and float(comp1) + float(comp2) < 10:
            print("Something wrong with your Comp files.")
        else:
            if not e:
                print("It's look you are ready to the submission!")
        scores.append(cur_score)
        if len(prob1) > 0:
            print(comp1_file, comp1)
        if len(prob2) > 0:
            print(comp2_file, comp2)
    scores = pd.DataFrame(scores)
    scores.to_csv('scores.csv')

def open_zip():
    errors = False
    zip_file_path = f"HW1_{ID1}_{ID2}.zip"
    if zip_file_path not in os.listdir():
        print(f"{zip_file_path} is not exists.")
        return True
    output_directory_path = "your_unzip_submission"
    unzip_directory(zip_file_path, output_directory_path)
    dir_files = os.listdir(output_directory_path)
    if len(dir_files) > 5:
        print("The submission contains redundant files.")
        errors = True
    comp_files = [f"comp_m1_{ID1}_{ID2}.wtag", f"comp_m2_{ID1}_{ID2}.wtag"]
    req_files = [f"report_{ID1}_{ID2}.pdf", "code", "generate_comp_tagged.py"] + comp_files
    for file in req_files:
        if file not in dir_files:
            print(f"{file} is not exists.")
            errors = True
    for file in comp_files:
        shutil.copy(os.path.join(output_directory_path, file), os.path.join(COMP_FILES_PATH, f"{ID1}_{ID2}"))
    return errors

if __name__ == '__main__':
    e = open_zip()
    calc_scores(e)
