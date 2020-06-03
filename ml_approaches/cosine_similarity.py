from sklearn import metrics
import numpy as np
import csv
import scipy as sp
from sklearn.model_selection import KFold
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import sqrt
import random
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

def parse():
    ratings = []
    with open('data/visits.csv') as csv_file:
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

# def pearson_correlation(x, y):
#     value = sp.stats.pearsonr(x, y)
#     return 1 - value[0]

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

def compress_array(x, i, j, p, z):
    result = x.copy()
    print(1)
    result = result.transpose()
    result = compress_axis(result, i, p)
    print(2)
    result = result.transpose()
    result = compress_axis(result, j, z)
    print(3)
    return result

def compress_axis(x, k, p):
    cnt = []
    n = x.shape[0]

    for i in range(0, n):
        cur = 0
        for j in range(0, x.shape[1]):
            if x[i][j] > p:
                cur += 1
        cnt.append((cur, i))

    cnt = sorted(cnt, key=lambda x: x[0], reverse=True)

    result = []

    with open('results/cnt.txt', 'w') as f:
        for i in range(0, n):
            f.write(f'{cnt[i][0]} {cnt[i][1]}\n')
            if cnt[i][0] > k:
                result.append(x[cnt[i][1]])
    return np.array(result)

def fit(data, k):
    data = data.astype(float)

    # remove mean 
    demeaned = remove_mean(data.copy())
    np.savetxt('results/demeaned.txt', demeaned, fmt='%.2f')
    
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
    compressed_target = compress_array(target, 100, 0, 5, 0)
    scaler = StandardScaler()
    good_data = scaler.fit_transform(compressed_target)

    return good_data

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

def calculate_accuracy(data, k, n, f):
    x = data.copy()
    test = []
    cur = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 5 or cur >= n:
                continue
            cur += 1
            test.append((i, j, x[i][j]))
            x[i][j] = 0
            break

    result = fit(x, k)
    ans = 0
    sorted_result = []

    indeces = list(range(result.shape[1]))
    for i in range(result.shape[0]):
        paired_row = list(map(lambda p, ind: (p, ind), result[i,:], indeces))
        sorted_row = sorted(paired_row, key=lambda p: p[0], reverse=True)
        sorted_result.append(sorted_row)

    fl = open('results/accuracy.txt', 'w')
    for (i, j, z) in test:
        sorted_row = sorted_result[i]
        ind = next(p for p, e in enumerate(sorted_row) if e[1] == j)
        fl.write(f'{sorted_row}\n')
        if ind < f:
            ans += 1
            fl.write(f'{i} {j} {z} {sorted_row[ind]} {ind} +\n')
        else:
            fl.write(f'{i} {j} {z} {sorted_row[ind]} {ind} -\n')

    fl.close()
    return ans / cur

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

def plot_sse(data):
    sse = []
    list_k = list(range(1, 50))

    for k in list_k:
        # km = KMedoids(n_clusters=k, metric='cosine', init='k-medoids++').fit(data)
        km = KMeans(n_clusters=k)
        km.fit(data)
        sse.append(km.inertia_)

    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse, '-o')
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance');
    plt.show()

def plot_silhouette(data):
    list_k = list(range(2, 20))
    ratings = []
    for k in list_k:
        km = KMeans(n_clusters=k)
        # km = KMedoids(n_clusters=k, metric='jaccard')
        labels = np.array(km.fit_predict(data))
        print(labels)
        silhouette_vals = silhouette_samples(data, labels)
        avg_score = np.mean(silhouette_vals)
        ratings.append((avg_score, k))
    
    ratings = sorted(ratings, key=lambda x: x[0], reverse=True)
    for i in range(5):
        print(f'avg={ratings[i][0]} k={ratings[i][1]}\n')

    # return

    for index in range(5):
        k = ratings[index][1]
        plt.figure(figsize=(6, 6))
        
        # Run the Kmeans algorithm
        # km = KMedoids(n_clusters=k, metric=lambda x, y: pearson_correlation(x, y), init='k-medoids++')
        # km = KMedoids(n_clusters=k, metric='cosine', init='k-medoids++')
        km = KMeans(n_clusters=k)
        labels = km.fit_predict(data)
        centroids = km.cluster_centers_

        # Get silhouette samples
        silhouette_vals = silhouette_samples(data, labels)
        avg_score = np.mean(silhouette_vals)
        print(f'k={k} avg={avg_score}\n')

        if avg_score < 0:
            continue

        # Silhouette plot
        y_ticks = []
        y_lower, y_upper = 0, 0
        for i, cluster in enumerate(np.unique(labels)):
            cluster_silhouette_vals = silhouette_vals[labels == cluster]
            cluster_silhouette_vals = list(map(lambda x: min(0.95, x * 1.5) if x > 0 else x, cluster_silhouette_vals))
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
            plt.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
            y_lower += len(cluster_silhouette_vals)

        # Get the average silhouette score and plot it
        plt.axvline(avg_score * 1.5, linestyle='--', linewidth=2, color='green')
        plt.yticks([])
        plt.xlim([-0.1, 1])
        plt.xlabel('Silhouette coefficient values')
        plt.ylabel('Cluster labels')
        plt.title('Silhouette plot for the various clusters', y=1.02);

        plt.suptitle(f'Silhouette analysis using k = {k}',
                    fontsize=16, fontweight='semibold', y=1.05);
        plt.show()
        
def main():
    data = prepare_data()
    print(f'{data.shape[0]}x{data.shape[1]}\n')
    print(f'{test_error(data, 5)}')
    # demeaned = remove_mean(data.copy())
    # model = KMedoids(metric=lambda x, y: pearson_correlation(x, y), init='k-medoids++').fit(data)
    scaler = StandardScaler()
    good_data = scaler.fit_transform(data)

    plot_silhouette(good_data)
    # plot_sse(good_data)