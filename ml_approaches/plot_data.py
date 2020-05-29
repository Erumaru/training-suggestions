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
    fitness_id = int(row[1])
    visit = int(row[2])

    if fitness_id in visits:
        visits[fitness_id] += visit
    else:
        visits[fitness_id] = visit

ratings_f = open('data/ratings.csv')
ratings = {}
rating_data = []
user_rating = {}
reader = csv.reader(ratings_f, delimiter=',')
for row in reader:
    user_id = int(row[0])
    fitness_id = int(row[1])
    rating = int(row[2])

    rating_data.append((user_id, fitness_id, rating))

for (user_id, fitness_id, rating) in rating_data:
    if user_id in user_rating:
        p = user_rating[user_id]
        user_rating[user_id] = (p[0] + rating, p[1] + 1)
    else:
        user_rating[user_id] = (rating, 1)

user_mean = {}
for key, value in user_rating.items():
    user_mean[key] = value[0] / value[1]

for (user_id, fitness_id, rating) in rating_data:
    local_rating = rating - user_mean[user_id]
    if fitness_id in ratings:
        p = ratings[fitness_id]
        ratings[fitness_id] = (p[0] + local_rating, p[1] + 1)
    else: 
        ratings[fitness_id] = (local_rating, 1)

ratings_f.close()
visits_f.close()


with open('data/studio_rating.csv', 'w') as f:
    f.write(f'id,average,total\n')
    for key in ratings.keys():
        r = ratings[key]
        f.write(f'{key},{r[0] / r[1]},{visits[key]}\n')
        


df = pd.read_csv('data/studio_rating.csv')
print(df.head())
df.columns = ['id', 'average', 'total']

df.plot(kind='scatter', x='total', y='average', logx=True, alpha=0.5, color='purple', edgecolor='')
plt.xlabel('Total visits (Gyms)')
plt.ylabel('Demeaned rating (Gyms)')
plt.show()