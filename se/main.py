from utils import *
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from deep_learning_models import Deep_model

def main():
    # shape of x, y = (690, 15), (690, )
    data_folds = preprocess_data('crx.data')

    model_names = ['SVM', 'Decision_Tree', 'Random_Forest', 'Ada_Boost', 'DNN', 'CNN']

    m = 'CNN'
    params_SVM = [
        {'kernel': 'rbf'},
        {'kernel': 'linear'},
        {'kernel': 'sigmoid'}
    ]
    params_DT = [
        {'criterion': 'gini', 'max_depth': 10},
        {'criterion': 'gini', 'max_depth': 25},
        {'criterion': 'gini', 'max_depth': 50},
        {'criterion': 'gini', 'max_depth': None},
        {'criterion': 'entropy', 'max_depth': 10},
        {'criterion': 'entropy', 'max_depth': 25},
        {'criterion': 'entropy', 'max_depth': 50},
        {'criterion': 'entropy', 'max_depth': None},
    ]
    params_RF = [
        {'n_estimators': 10},
        {'n_estimators': 50},
        {'n_estimators': 100},
        {'n_estimators': 300},
    ]
    params_AB = [
        {'n_estimators': 10},
        {'n_estimators': 50},
        {'n_estimators': 100},
        {'n_estimators': 300},
    ]
    params_CNN = [
        {'epoch': 50, 'lr': 1e-1},
        {'epoch': 50, 'lr': 1e-2},
        {'epoch': 50, 'lr': 1e-3},
        {'epoch': 100, 'lr': 1e-1},
        {'epoch': 100, 'lr': 1e-2},
        {'epoch': 100, 'lr': 1e-3},
        {'epoch': 300, 'lr': 1e-1},
        {'epoch': 300, 'lr': 1e-2},
        {'epoch': 300, 'lr': 1e-3},
    ]
    for p in params_CNN:

        f1_arr = []
        pre_arr = []
        rec_arr = []
        acc_arr = []
        for i in range(10):

            training_data = data_folds[:i] + data_folds[i+1:]
            test_data = data_folds[i]

            y_, y = classifier(training_data, test_data, m, p)
            f1, pre, rec, acc = eval(y, y_)
            f1_arr.append(f1)
            pre_arr.append(pre)
            rec_arr.append(rec)
            acc_arr.append(acc)
        print('model "{}"'.format(m))
        print('param: {}'.format(p))
        print('acc: {} precision: {} recall: {} f1: {}'.\
              format(round(np.mean(acc_arr), 4), round(np.mean(pre_arr), 4),
                     round(np.mean(rec_arr), 4), round(np.mean(f1_arr), 4)))

def classifier(training_data, testing_data, model_name, kwargs):

    train_x = []
    train_y = []
    for td in training_data:
        train_x.append(td[0])
        train_y.append(td[1])

    train_x = np.reshape(train_x, [-1, np.shape(train_x)[-1]])
    train_y = np.reshape(train_y, [-1, np.shape(train_y)[-1]])

    test_x, test_y = testing_data

    if model_name not in ['DNN', 'CNN']:
        train_y = train_y.reshape(-1)
        test_y = test_y.reshape(-1)

    if model_name == 'SVM':
        clf = SVC(gamma='auto', **kwargs)
    elif model_name == 'Decision_Tree':
        clf = DecisionTreeClassifier(**kwargs)
    elif model_name == 'Random_Forest':
        clf = RandomForestClassifier(**kwargs)
    elif model_name == 'Ada_Boost':
        clf = AdaBoostClassifier(**kwargs)
    elif model_name == 'DNN':
        clf = Deep_model('DNN', train_x.shape[-1], 64, **kwargs)
    elif model_name == 'CNN':
        clf = Deep_model('CNN', 1, 32, **kwargs)
    else:
        print('No such model')
        exit(-2)
    clf.fit(train_x, train_y)

    return clf.predict(test_x), test_y

def eval(y, y_):

    f1 = f1_score(y, y_)
    precision = precision_score(y, y_)
    recall = recall_score(y, y_)
    acc = accuracy_score(y, y_)
    return f1, precision, recall, acc


if __name__ == '__main__':
    main()