import numpy as np
import pandas as pd
from scipy.signal import medfilt
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from utils.helper import any_transform
from sklearn.base import clone

class RegressionEnsamble(BaseEstimator):
  """
  Regression model wrapper for different ML models like SVM or LR.
  Uses regression for node pressure prediction which then gets used
  together with actual node pressure to classify. 
  This happens for every node which creates a simulated pressure sensor network.
  Uses median filter for noise free classification.
  """

  def __init__(self, model, medfilt_kernel_size=5, nodes=['10','11','12','13','21','22','23','31','32'], **model_params):
    """Create the model.

    Args:
      model (SKLearn regression model): The base estimator.
      medfilt_kernel_size (int): The kernel size for the median filter.
      nodes (list of nodes): Nodes used for regression ensamble.
    """
    self.models = {}
    for node in nodes:
      self.models[node] = clone(model)
      if model_params:
        self.models[node].set_params(**model_params)

    self.nodes = nodes
    self.medfilt_kernel_size = medfilt_kernel_size

  def fit(self, X, y):
    """Trains the model.

    Args:
      X (list of pandas dataframe time series): Multiple simulations.
      y (list of numpy arrays): True labels for every time point in every simulation.
    """

    # Concatinate the simulations to train everything at once.
    if type(X) == list:
      X_concat = pd.concat(X)
      X_concat.reset_index(drop=True, inplace=True)
    else:
      raise TypeError('X must be list of pd.DataFrame.')
    y_concat = np.concatenate(y)

    # Train the regression on just non-leakage scenarios
    data_regr = X_concat[y_concat == 0]

    # Go over every virtual node
    X_regr_pred = pd.DataFrame()
    for node in self.models:

      # Filter features for regression
      features = list(X_concat.columns)
      features.remove(node)
      X_regr = data_regr[features]
      y_regr = data_regr[node]

      # Train regression model
      self.models[node].fit(X_regr, y_regr)

      # Pressure prediction for Threshold calculation
      X_regr_pred[node] = self.models[node].predict(X_concat[features])
      
    differences = X_concat[list(self.models)] - X_regr_pred

    return differences

    #self.threshold = differences[y_concat == 0].quantile(.0)[0]

    return self

  def predict(self, X):
    """Predict labels for every time point of every time series inputted.

    Args:
      X (list of pandas dataframe time series): Multiple time series.
    
    Returns:
      preds: List of Numpy arrays containing the labels for every time point.
    """
    preds = []
    for X_single in X:

      # Go over every virtual node
      X_regr_pred = pd.DataFrame()
      for node in self.models:
  
        # Exclude current node for regression
        features = list(X_single.columns)
        features.remove(node)
  
        # Pressure prediction for Threshold calculation
        X_regr_pred[node] = self.models[node].predict(X_single[features])
        
      differences = X_single[list(self.models)] - X_regr_pred
      
      # ?
      #is_under_threshold = np.array((differences < self.threshold)['12'])
      #preds.append(is_under_threshold.astype(int))
    
    return preds

  def score(self, X, y):
    """Implements standard accuracy score.

    Args:
      X (list of pandas dataframe time series): Multiple time series.
      y (list of numpy arrays): True labels for every time point in every simulation.
    
    Returns:
      score: Accuracy score.
    """
    return accuracy_score(*any_transform(y, self.predict(X)))

  def get_params(self, deep=True):
    params = self.models[self.nodes[0]].get_params(deep)
    params['model'] = self.model
    params['nodes'] = self.nodes
    params['medfilt_kernel_size'] = self.medfilt_kernel_size
    return params
  
  def set_params(self, **params):
    if 'model' in params:
      self.model = params['model']
      del params['model']
    if 'nodes' in params:
      self.nodes = params['nodes']
      del params['nodes']
    if 'medfilt_kernel_size' in params:
      self.medfilt_kernel_size = params['medfilt_kernel_size']
      del params['medfilt_kernel_size']
    for node in self.nodes:
      self.models[node].set_params(**params)
    return self
