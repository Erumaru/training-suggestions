import numpy as np
import csv

def read_data():
    X = []
    with open('data/1fit.data') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            X.append(row)
    return np.array(X)


X = read_data()
print(X)
from sklearn.decomposition import NMF
model = NMF(n_components=2, init='random', random_state=0)
model.fit(X)