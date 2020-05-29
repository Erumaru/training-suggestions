import os
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import Dataset
from surprise import accuracy
from surprise import BaselineOnly
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
import models

# global model variable 
algo = SVDpp(n_factors=100, verbose=True, n_epochs=100)

def train():
    # load dataset
    file_path = os.path.expanduser('data/ratings.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)
    training_set = data.build_full_trainset()

    # train model
    algo.fit(training_set)

    # test 
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=-1)

train()
# write_results(predictions, zhibek)
