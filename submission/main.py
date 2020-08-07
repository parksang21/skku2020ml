from data import get_data
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from models import *

MODELS = ['CNN', 'SVM', 'Decision_Tree', 'Random_Forest', 'Ada_Boost']

PARMAS = {
    'SVM': [
        {'kernel': 'rbf'},
        {'kernel': 'linear'},
        {'kernel': 'sigmoid'}
    ],
    'Decision_Tree': [
        {'criterion': 'gini', 'max_depth': 10},
        {'criterion': 'gini', 'max_depth': 25},
        {'criterion': 'gini', 'max_depth': 50},
        {'criterion': 'gini', 'max_depth': None},
        {'criterion': 'entropy', 'max_depth': 10},
        {'criterion': 'entropy', 'max_depth': 25},
        {'criterion': 'entropy', 'max_depth': 50},
        {'criterion': 'entropy', 'max_depth': None},
    ],
    'Random_Forest': [
        {'n_estimators': 10},
        {'n_estimators': 50},
        {'n_estimators': 100},
        {'n_estimators': 300},
    ],
    'Ada_Boost': [
        {'n_estimators': 10},
        {'n_estimators': 50},
        {'n_estimators': 100},
        {'n_estimators': 300},
    ],
    'CNN': [
        {'epoch': 50, 'lr': 1e-1},
        {'epoch': 50, 'lr': 1e-2},
        {'epoch': 50, 'lr': 1e-3},
        {'epoch': 100, 'lr': 1e-1},
        {'epoch': 100, 'lr': 1e-2},
        {'epoch': 100, 'lr': 1e-3},
        {'epoch': 300, 'lr': 1e-1},
        {'epoch': 300, 'lr': 1e-2},
        {'epoch': 300, 'lr': 1e-3},
    ],
}


def evaluation(gt, prediction):
    return f1_score(gt, prediction), precision_score(gt, prediction), recall_score(gt, prediction), accuracy_score(gt, prediction)

if __name__ == "__main__":
    data, label = get_data('datasets/ionosphere.data')
    skf = StratifiedKFold(n_splits=10, shuffle=True)

    for m in MODELS:
        for p in PARMAS[m]:
            f1_scores, precisions, recalls, accuracies = [], [], [], []
            for i in range(10):
                for train_index, test_index in skf.split(data, label):
                    train = [[data[idx] for idx in train_index],
                             [label[idx] for idx in train_index]]
                    test = [[data[idx] for idx in test_index],
                            [label[idx] for idx in test_index]]
                    # print(len(train_index))
                    prediction, gt, model = classifier(train, test, m, p)

                    f1, pre, rec, acc = evaluation(gt, prediction)
                    f1_scores.append(f1)
                    precisions.append(pre)
                    recalls.append(rec)
                    accuracies.append(acc)
            print(f'{m}, params: {p}', end='\t\t\t\t\t')
            print('acc: {} precision: {}, recall: {}, f1: {}'.format(
                round(np.mean(accuracies), 4), round(np.mean(precisions), 4),
                round(np.mean(recalls), 4), round(np.mean(f1_scores), 4)
            ))
