
import csv
from sklearn import preprocessing

import os

def get_data(path):
    print(f"Preparing Data\nPath : {path}")
    data = []
    label = []

    if os.path.exists(path) is False:
        print(f"path {path} is not valid")
        import requests
        response = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases' +
         '/ionosphere/ionosphere.data')

    with open(path) as data_file:
        raw_data = csv.reader(data_file, delimiter=',')
        assert raw_data is not None, 'Data Loading Error'

        for row in raw_data:
            data.append([float(r)for r in row[:-1]])
            label.append(0 if row[-1] == 'b' else 1)
        if len(data) != len(label):
            print('Data and Label Does Not Match')
        else:
            print('Data is successfully loaded')

    return preprocessing.scale(data), label



