from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

a = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]])

#総当たりで２つのarrayそれぞれの類似度を出す
print(cosine_similarity(a))