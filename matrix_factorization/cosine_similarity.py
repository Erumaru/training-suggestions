from sklearn import metrics
import numpy as np
import csv
import scipy as sp
from sklearn.model_selection import KFold
from math import sqrt
import random

def parse():
    ratings = []
    with open('data/user_studio_visit.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for row in reader:
            ratings.append((int(row[0]), int(row[1]), int(row[2])))
    return ratings

def pearson_correlation(x):
    rows = x.shape[0]

    f = open('results/super_sim.txt', 'w')
    # create empty array with zero
    result = np.zeros((rows, rows), float)
    for i in range(0, rows):
        for j in range(0, rows):
            value = sp.stats.pearsonr(x[i], x[j])
            result[i][j] = value[0]
            if value[0] > 0.5 and i != j:
                f.write(f'val: {value[0]}\np: {value[1]}\n{x[i]}\n{x[j]}\n')

    f.close()
    return result

def remove_n(x, n):
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            if x[i][j] != 0:
                x[i][j] -= n

    return x


def compress_axis(x):
    cnt = []
    n = x.shape[0]

    for i in range(0, n):
        cur = 0
        for j in range(0, x.shape[1]):
            if x[i][j] != 0:
                cur += 1
        cnt.append((cur, i))

    cnt = sorted(cnt, key=lambda x: x[0], reverse=True)

    result = []

    f = open('results/cnt.txt', 'w')
    for i in range(0, n):
        f.write(f'{cnt[i][0]} {cnt[i][1]}\n')
        if cnt[i][0] > 9:
            result.append(x[cnt[i][1]])

    f.close()

    x = np.array(result)
    x = x.transpose()
    x = x[~np.all(x == 0, axis=1)]
    x = x.transpose()
    return x

def fit(data, k):
    data = data.astype(float)
    # find row means
    # means = np.nanmean(np.where(data != 0, data, np.nan), 1)
    mean = np.nanmean(np.where(data != 0, data, np.nan))
    # print(mean)

    # remove mean 
    # data = remove_n(data, mean)
    
    # remove 3
    # data = remove_n(data, 3)

    # find pearson correlation
    similarities = pearson_correlation(data)

    # find cosine similarity
    # similarities = metrics.pairwise.cosine_similarity(data)
    
    result = data.copy().astype(float)

    file = open('results/sim.txt', 'w')
    for x in range(0, data.shape[0]):
        user_sim = similarities[:,x]
        indeces = list(range(0, len(user_sim)))
        paired_user_sim = list(map(lambda p, index: (p, index), user_sim, indeces))
        for y in range(0, data.shape[1]):
            value = data[x][y]
            if value == 0:
                filtered_user_sim = list(filter(lambda p: data[p[1]][y] != 0, paired_user_sim))
                sorted_user_sim = sorted(filtered_user_sim, key=lambda p: p[0], reverse=True)
                cur = 0
                cur_sim = 0
                for (sim, user_id) in sorted_user_sim[:k]:
                    cur += sim * data[user_id][y]
                    cur_sim += sim
                    file.write(f"{x} {y} {sim} {data[user_id][y]}\n")
                file.write(f"{x} {y} cur: {cur}\n")
                if cur > 0:
                    result[x][y] = cur / cur_sim
            # result[x][y] += 3
            # result[x][y] += mean
    file.close()
    np.savetxt('results/cosine_result.txt', result, fmt='%.2f')
    return result

def prepare_data():
    ratings = parse()
    data = np.array(ratings)
    n = data[:, 0].max()
    m = data[:, 1].max()

    # create empty array with zero
    target = np.zeros((n+1, m+1), data.dtype)
    target[data[:, 0], data[:, 1]] = data[:, 2]

    # transpose
    # compress coordinates
    compressed_target = compress_axis(target)

    return compressed_target

def calculate_error(x, y):
    result = 0
    n = 0
    f = open('results/error.txt', 'w')
    for i in range(0, y.shape[0]):
        for j in range(0, y.shape[1]):
            if y[i][j] == 0:
                continue
            f.write(f'{x[i][j]}\t{y[i][j]}\n')
            result += (x[i][j] - y[i][j]) ** 2
            n += 1

    f.close()
    return sqrt(result / n)


def test_error(x, k):
    train = x.copy()
    test = np.zeros((x.shape[0], x.shape[1]), dtype=x.dtype)

    for i in range(0, x.shape[0]):
        cur = []
        for j in range(0, x.shape[1]):
            if x[i][j] != 0: 
                cur.append(j)
        rnd = random.choice(cur)
        train[i][rnd] = 0
        test[i][rnd] = x[i][rnd]

    result = fit(train, k)
    error = calculate_error(result, test)

    return error
    
data = prepare_data()


for i in range(2, 3):
    print(f'k={i} rmse={test_error(data, i)}')