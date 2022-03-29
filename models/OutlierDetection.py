import numpy as np
import pandas as pd
from scipy.signal import medfilt
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from utils.helper import any_transform

class OutlierDetectionModel(BaseEstimator):
  """
  Outlier Detection model wrapper for different ML models like IsoForest.
  Uses median filter for noise free classification.
  """

  def __init__(self, model, medfilt_kernel_size=5, **model_params):
    """Create the model.

    Args:
      model (SKLearn classificator): The base estimator.
      medfilt_kernel_size (int): The kernel size for the median filter.
    """
    self.model = model
    self.medfilt_kernel_size = medfilt_kernel_size
    if model_params:
      self.model.set_params(**model_params)

  def fit(self, X, y=None):
    """Trains the model.

    Args:
      X (list of pandas dataframe time series): Multiple simulations.
      y: For compatibility, will be ignored.
    """

    # Concatinate the simulations to train everything at once.
    if type(X) == list:
      X_concat = pd.concat(X)
      X_concat.reset_index(drop=True, inplace=True)
    else:
      X_concat = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))

    # Train the model.
    self.model.fit(X_concat)

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
      pred = self.model.predict(X_single)*(-.5)+.5
      preds.append(medfilt(pred, self.medfilt_kernel_size))
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
