import numpy as np
import csv 
import scipy.sparse as spr
import studios
import models

import nimfa

def read_data():
    rows = []
    cols = []
    data = []
    with open('data/1fit.data') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            rows.append(int(row[0]))
            cols.append(int(row[1]))
            data.append(int(row[2]))
    return rows, cols, data

r, c, d = read_data()

V = spr.csr_matrix((d, (r, c)))
print('Target:\n%s' % V.todense())

nmf = nimfa.Nmf(V, max_iter=200, rank=2, update='euclidean', objective='fro')
nmf_fit = nmf()

W = nmf_fit.basis()
print('Basis matrix:\n%s' % W.todense())

H = nmf_fit.coef()
print('Mixture matrix:\n%s' % H.todense())

print('Euclidean distance: %5.3f' % nmf_fit.distance(metric='euclidean'))

sm = nmf_fit.summary()
result = np.dot(W.todense(), H.todense())
print('Sparseness Basis: %5.3f  Mixture: %5.3f' % (sm['sparseness'][0], sm['sparseness'][1]))
print('Iterations: %d' % sm['n_iter'])
print('Target estimate:\n%s' % result)
print(len(result))

studios.parse()


def find_rating_of_user(id):
    predictions = []
    for studio in studios.studios:
        rating = result[int(id)][int(studio.id)]
        predictions.append(models.Prediction(studio, rating))
    
    return predictions

def write_results(predictions, user):
    # sort prediction by rating
    sorted_predictions = reversed(sorted(predictions, key=lambda prediction: prediction.rating))

    text_file = open("result_nimfa.txt", "w")
    for id, p in enumerate(sorted_predictions):
        row = f'position: {id},\tuser_id: {user},\tstudio_id: {p.studio.id},\tstudio_name: {p.studio.name},\trating: {p.rating}\n'
        text_file.write(row)
    text_file.close()

abzal = str(132)
alisher = str(73)
zhibek = str(1172)
predictions = find_rating_of_user(zhibek)
write_results(predictions, zhibek)