from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
from sklearn.decomposition import PCA
import cosine_similarity

data = cosine_similarity.prepare_data()
print(f'{data.shape[0]}x{data.shape[1]}\n')
pca = PCA()
pca.fit(data)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)