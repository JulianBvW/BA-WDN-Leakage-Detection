import numpy as np
import pandas as pd
from scipy.signal import medfilt
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from utils.helper import any_transform

class RegressionModel(BaseEstimator):
  """
  Regression model wrapper for different ML models like SVM or LR.
  Uses Regressor for node pressure prediction which then gets used
  together with actual node pressure to classify. 
  Uses median filter for noise free classification.
  """

  def __init__(self, model, medfilt_kernel_size=5, node_target='12', node_ensamble=['10', '11','13','21','22','23','31','32'], **model_params):
    """Create the model.

    Args:
      model (SKLearn regressor): The base estimator.
      medfilt_kernel_size (int): The kernel size for the median filter.
      node_target (node): The node which value should be regressed.
      node_ensamble (list of nodes): Nodes used for regression.
    """
    self.model = model
    self.node_target = node_target
    self.node_ensamble = node_ensamble
    self.medfilt_kernel_size = medfilt_kernel_size
    if model_params:
      self.model.set_params(**model_params)

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

    # Filter leakages
    data_regr = X_concat[y_concat == 0]
    X_regr = data_regr[self.node_ensamble] # TODO: time of day?
    y_regr = data_regr[self.node_target]

    # Train Regressor
    self.model.fit(X_regr, y_regr)

    # Set Threshold
    y_pred = self.model.predict(X_concat[self.node_ensamble]) # TODO: time of day?
    differences = pd.DataFrame(X_concat['12'] - y_pred)
    self.threshold = differences[y_concat == 0].quantile(.0)[0]

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
      X_single_regr = X_single[self.node_ensamble] # TODO: time of day?
      y_single_true = X_single[self.node_target]
      y_single_pred = self.model.predict(X_single_regr)
      differences = pd.DataFrame(y_single_true - y_single_pred)
      is_under_threshold = np.array((differences < self.threshold)['12'])
      preds.append(is_under_threshold.astype(int))
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
    params = self.model.get_params(deep)
    params['model'] = self.model
    params['medfilt_kernel_size'] = self.medfilt_kernel_size
    return params
  
  def set_params(self, **params):
    if 'model' in params:
      self.model = params['model']
      del params['model']
    if 'medfilt_kernel_size' in params:
      self.medfilt_kernel_size = params['medfilt_kernel_size']
      del params['medfilt_kernel_size']
    self.model.set_params(**params)
    return self
