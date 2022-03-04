import numpy as np

def any_transform(a, b):
  any = lambda l: int(sum(l) > 0)
  
  a = list(map(any, a))
  b = list(map(any, b))

  return a, b

def shuffle_data(X, y):
  idxs = np.arange(len(y))
  np.random.shuffle(idxs)
  return X[idxs], y[idxs]
