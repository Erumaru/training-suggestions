from sklearn import metrics
import numpy as np
import csv
import scipy as sp
from sklearn.model_selection import KFold
from math import sqrt
import random

def parse():
    ratings = []
    with open('data/ratings.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
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

def remove_mean(x):
    for i in range(0, x.shape[0]):
        sum = 0
        cnt = 0
        for j in range(0, x.shape[1]):
            if x[i][j] != 0: 
                sum += x[i][j]
                cnt += 1

        row_mean = sum / cnt
        print(f'{row_mean} {sum} {cnt}')
        for j in range(0, x.shape[1]):
            if x[i][j] != 0:
                x[i][j] -= row_mean
        

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

    with open('results/cnt.txt', 'w') as f:
        for i in range(0, n):
            f.write(f'{cnt[i][0]} {cnt[i][1]}\n')
            if cnt[i][0] > 2:
                result.append(x[cnt[i][1]])

    x = np.array(result)
    x = x.transpose()
    x = x[~np.all(x == 0, axis=1)]
    x = x.transpose()
    return x

def fit(data, k):
    data = data.astype(float)

    # remove mean 
    demeaned = remove_mean(data.copy())
    np.savetxt('results/demeaned.txt', demeaned, fmt='%.2f')
    
    # remove 3
    # data = remove_n(data, 3)

    # find pearson correlation
    # similarities = pearson_correlation(demeaned)

    # find cosine similarity
    similarities = metrics.pairwise.cosine_similarity(data)
    
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
                if cur > 0:
                    result[x][y] = cur / cur_sim
                    file.write(f"{x} {y} cur: {cur / cur_sim}\n")
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

def calculate_accuracy(x, k, n):
    result = fit(x, k)
    cnt = 0
    value = 0
    fl = open('results/accuracy.txt', 'w')
    for i in range(result.shape[0]):
        indeces = list(range(result.shape[1]))
        pred_paired_row = list(map(lambda p, index: (p, index), result[i,:], indeces))
        pred_sorted_row = sorted(pred_paired_row, key=lambda p: p[0], reverse=True)
        paired_row = list(map(lambda p, index: (p, index), x[i,:], indeces))
        sorted_row = sorted(paired_row, key=lambda p: p[0], reverse=True)
        fl.write(f'{list(map(lambda p: int(p[1]), sorted_row))}\n')
        fl.write(f'{list(map(lambda p: int(p[0]), sorted_row))}\n')
        fl.write(f'{list(map(lambda p: int(p[1]), pred_sorted_row))}\n')
        fl.write(f'{list(map(lambda p: int(p[0]), pred_sorted_row))}\n')
        for j in range(n):
            val = pred_sorted_row[j][0] 
            stud_ind = pred_sorted_row[j][1] 
            if x[i][stud_ind] == 0:
                continue
            cnt += 1

            ind = next(p for p, e in enumerate(sorted_row) if e[1] == stud_ind)
            if ind < n:
                value += 1
                fl.write(f'{i} {stud_ind} {val} {ind} +\n')
            else:
                fl.write(f'{i} {stud_ind} {val} {ind} -\n')
    fl.close()
    return value / cnt


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


print(calculate_accuracy(data, 10, 10))
# print(f'k=10 rmse={test_error(data, 10)}')