import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
sns.set_style('darkgrid')

visits_f = open('data/visits.csv')
visits = {}
reader = csv.reader(visits_f, delimiter=',')
for row in reader:
    user_id = int(row[0])
    fitness_id = int(row[1])
    visit = int(row[2])

    if fitness_id in visits:
        p = visits[fitness_id]
        visits[fitness_id] = (p[0] + visit, p[1] + 1)
    else:
        visits[fitness_id] = (visit, 1)

ratings_f = open('data/ratings.csv')
ratings = {}
rating_data = []
user_rating = {}
reader = csv.reader(ratings_f, delimiter=',')
for row in reader:
    user_id = int(row[0])
    fitness_id = int(row[1])
    rating = int(row[2])

    if fitness_id in ratings:
        p = ratings[fitness_id]
        ratings[fitness_id] = (p[0] + rating, p[1] + 1)
    else:
        ratings[fitness_id] = (rating, 1)

ratings_f.close()
visits_f.close()

with open('data/studio_rating.csv', 'w') as f:
    f.write(f'rating, visits\n')
    for key in ratings.keys():
        v = visits[key]
        r = ratings[key]
        f.write(f'{r[0] / r[1]},{v[0] / v[1]}\n')
        

def main():
    df = pd.read_csv('data/studio_rating.csv')
    print(df.head())
    df.columns = ['rating', 'visit']

    df.plot(kind='scatter', x='visit', y='rating', logx=True, alpha=0.5, color='purple', edgecolor='')
    plt.ylabel('Rating (Gyms)')
    plt.xlabel('Visits (Gyms)')
    plt.show()

main()