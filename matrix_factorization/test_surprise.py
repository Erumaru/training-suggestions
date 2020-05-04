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
import studios
import models

# global model variable 
algo = SVDpp(init_mean=0)

def train():
    # load dataset
    file_path = os.path.expanduser('data/1fit.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)
    training_set = data.build_full_trainset()

    # train model
    algo.fit(training_set)

    # test 
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, verbose=True, n_jobs=-1)


def init_studios():
    # init studios 
    studios.parse()

def find_rating_of_user(id):
    predictions = []
    for studio in studios.studios:
        rating = algo.predict(id, studio.id)
        predictions.append(models.Prediction(studio, rating.est))
    
    return predictions

def write_results(predictions, user):
    # sort prediction by rating
    sorted_predictions = reversed(sorted(predictions, key=lambda prediction: prediction.rating))

    text_file = open("result.txt", "w")
    for id, p in enumerate(sorted_predictions):
        row = f'position: {id},\tuser_id: {user},\tstudio_id: {p.studio.id},\tstudio_name: {p.studio.name},\trating: {p.rating}\n'
        text_file.write(row)
    text_file.close()

# MAIN
abzal = str(132)
alisher = str(73)
zhibek = str(1172)
train()
init_studios()
predictions = find_rating_of_user(zhibek)
write_results(predictions, zhibek)
