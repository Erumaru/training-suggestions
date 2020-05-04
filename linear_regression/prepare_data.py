import models
import storage
import geopy.distance
from sklearn import preprocessing
from numpy import loadtxt, savetxt

# methods
def home_distance(key):
    user = storage.user_by_id[key[0]]
    studio = storage.studio_by_id[key[1]]

    c1 = (user.home_latitude, user.home_longitude)
    c2 = (studio.latitude, studio.longitude)

    return geopy.distance.distance(c1, c2).km

def work_distance(key):
    user = storage.user_by_id[key[0]]
    studio = storage.studio_by_id[key[1]]

    c1 = (user.work_latitude, user.work_longitude)
    c2 = (studio.latitude, studio.longitude)

    return geopy.distance.distance(c1, c2).km

def avg(lst): 
    return sum(lst) / len(lst) 

# parse data
storage.parse()

u_s_visit = {}

for visit in storage.visits:
    key = (visit.user_id, visit.fitness_id)
    if u_s_visit.__contains__(key):
        u_s_visit[key] += 1
    else:
        u_s_visit[key] = 1

max_rating = max(list(map(lambda x: x.rating, storage.studios)))
max_rating_count = max(list(map(lambda x: x.rating_count, storage.studios)))
max_one_visit_price = max(list(map(lambda x: x.one_visit_price, storage.studios)))
max_unlimited_price = max(list(map(lambda x: x.unlimited_price, storage.studios)))
max_distance = 30

with open('data/user_studio_visit.csv', 'w') as file: 
    for key, value in u_s_visit.items():
        user_id = int(key[0])
        studio = storage.studio_by_id[key[1]]

        rating = studio.rating / max_rating 
        rating_count = studio.rating_count / max_rating_count 
        one_visit_price = studio.one_visit_price / max_one_visit_price 
        unlimited_price = studio.unlimited_price / max_unlimited_price 
        h_dist = min(home_distance(key) / max_distance, 1.0) 
        w_dist = min(work_distance(key) / max_distance, 1.0) * 1000

        file.write(f'{user_id} {key[1]} {value}\n')

# with open('formatted_data_y.txt', 'w') as file:
#     for _, value in u_s_visit.items():
#         file.write(f'{value}\n')


# x = loadtxt('formatted_data_x.txt')
# scaled_x = preprocessing.scale(x)
# savetxt('formatted_data_x.txt', scaled_x)