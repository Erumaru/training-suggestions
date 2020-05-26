from fastai.collab import *
import pandas as pd
import numpy as np

# read data
ratings = pd.read_csv('data/abzal.csv')
print(ratings.head())

# data from file
data = CollabDataBunch.from_df(ratings=ratings, valid_pct=0.2)

# create learner
learn = collab_learner(data, n_factors=50, y_range=(0.,5.))

# fit
learn.fit(epochs=30) 

learn.show_results()
