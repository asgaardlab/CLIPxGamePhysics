import faiss
import numpy as np


class FaissNeighbors:
  def __init__(self):
    self.index = None
    self.y = None

  def fit(self, X, y):
    self.index = faiss.IndexFlatL2(X.shape[1])
    self.index.add(X.astype(np.float32))
    self.y = y
  
  def get_distances_and_indices(self, X, top_K=1000):
    distances, indices = self.index.search(X.astype(np.float32), k=top_K)
    return np.copy(distances), np.copy(indices), np.copy(self.y[indices])
  
  def get_nearest_labels(self, X, top_K=1000):
    distances, indices = self.index.search(X.astype(np.float32), k=top_K)
    return np.copy(self.y[indices])


class FaissCosineNeighbors:
  def __init__(self):
    self.cindex = None
    self.y = None

  def fit(self, X, y):
    self.cindex = faiss.index_factory(X.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    X = np.copy(X)
    X = X.astype(np.float32)
    faiss.normalize_L2(X)
    self.cindex.add(X)
    self.y = y
  
  def get_distances_and_indices(self, Q, topK):
    Q = np.copy(Q)
    faiss.normalize_L2(Q)
    distances, indices = self.cindex.search(Q.astype(np.float32), k=topK)
    return np.copy(distances), np.copy(indices), np.copy(self.y[indices])
  
  def get_nearest_labels(self, Q, topK=1000):
    Q = np.copy(Q)
    faiss.normalize_L2(Q)
    distances, indices = self.cindex.search(Q.astype(np.float32), k=topK)
    return np.copy(self.y[indices])