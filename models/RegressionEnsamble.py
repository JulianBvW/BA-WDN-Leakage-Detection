import numpy as np
import pandas as pd
from pexpect.spawnbase import _NullCoder
from scipy.signal import medfilt
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from utils.helper import any_transform
from sklearn.base import clone
from tqdm import tqdm

class RegressionEnsamble(BaseEstimator):
  """
  Regression model wrapper for different ML models like SVM or LR.
  Uses regression for node pressure prediction which then gets used
  together with actual node pressure to classify.
  This happens for every node which creates a simulated pressure sensor network.
  Uses median filter for noise free classification.
  """

  def __init__(self, model, th_mode='simple', th_multiplier=1.5, th_majority=0.9, medfilt_kernel_size=5, nodes=['10','11','12','13','21','22','23','31','32'], **model_params):
    """Create the model.

    Args:
      model (SKLearn regression model): The base estimator.
      th_mode ('simple' or 'daytime'): Mode for threshold handling.
      th_multiplier (float): Increases the threshold (maximal error while fit).
      th_majority (float): Percentage of nodes that are needed for classification voting.
      medfilt_kernel_size (int): The kernel size for the median filter.
      nodes (list of nodes): Nodes used for regression ensamble.
    """
    self.models = {}
    for node in nodes:
      self.models[node] = clone(model)
      if model_params:
        self.models[node].set_params(**model_params)
    
    self.th_mode = th_mode
    self.th_multiplier = th_multiplier
    self.th_majority = th_majority
    self.nodes = nodes
    self.medfilt_kernel_size = medfilt_kernel_size

  def fit(self, X, y, verbose=False):
    """Trains the model.

    Args:
      X (list of pandas dataframe time series): Multiple simulations.
      y (list of numpy arrays): True labels for every time point in every simulation.
      verbose (bool): If you want to use tqdm.
    """

    nodes = self.nodes
    if verbose:
      nodes = tqdm(self.nodes)

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
    for node in nodes:

      # Filter features for regression
      features = list(X_concat.columns)
      features.remove(node)
      X_regr = data_regr[features]
      y_regr = data_regr[node]

      # Train regression model
      self.models[node].fit(X_regr, y_regr)

      # Pressure prediction for Threshold calculation
      X_regr_pred[node] = self.models[node].predict(X_concat[features])
      
    differences = X_concat[nodes] - X_regr_pred

    if self.th_mode == 'simple':
      self.thresholds = abs(differences[y_concat == 0]).max() * self.th_multiplier
    else:
      differences['hour of the day'] = X_concat['hour of the day']
      self.thresholds = abs(differences[y_concat == 0]).groupby(['hour of the day']).max() * self.th_multiplier

    return self

  def predict(self, X, verbose=False):
    """Predict labels for every time point of every time series inputted.

    Args:
      X (list of pandas dataframe time series): Multiple time series.
      verbose (bool): If you want to use tqdm.
    
    Returns:
      preds: List of Numpy arrays containing the labels for every time point.
    """

    if verbose:
      X = tqdm(X)

    preds = []
    for X_single in X:

      # Go over every virtual node
      X_regr_pred = pd.DataFrame()
      for node in self.nodes:
  
        # Exclude current node for regression
        features = list(X_single.columns)
        features.remove(node)
  
        # Pressure prediction for Threshold calculation
        X_regr_pred[node] = self.models[node].predict(X_single[features])
        
      differences = X_single[self.nodes] - X_regr_pred
      min_voters = len(self.nodes) * self.th_majority

      if self.th_mode == 'simple':
        pred = (differences > self.thresholds).sum(axis=1)
        pred = (pred >= min_voters).astype(int)
      else:
        by_daytimes = []
        for daytime in X_single['hour of the day'].unique():
          by_daytime = differences[X_single['hour of the day'] == daytime]
          by_daytimes.append(by_daytime > self.thresholds.loc[daytime])
        pred = pd.concat(by_daytimes).sort_index().sum(axis=1)
        pred = (pred >= min_voters).astype(int)

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
    params = self.models[self.nodes[0]].get_params(deep)
    params['th_mode'] = self.th_mode
    params['th_multiplier'] = self.th_multiplier
    params['th_majority'] = self.th_majority
    params['model'] = self.models[self.nodes[0]]
    params['nodes'] = self.nodes
    params['medfilt_kernel_size'] = self.medfilt_kernel_size
    return params
  
  def set_params(self, **params):
    if 'nodes' in params:
      self.nodes = params['nodes']
      del params['nodes']
    if 'th_mode' in params:
      self.th_mode = params['th_mode']
      del params['th_mode']
    if 'th_multiplier' in params:
      self.th_multiplier = params['th_multiplier']
      del params['th_multiplier']
    if 'th_majority' in params:
      self.th_majority = params['th_majority']
      del params['th_majority']
    if 'model' in params:
      for node in self.nodes:
        self.models[node] = clone(params['model'])
      del params['model']
    if 'medfilt_kernel_size' in params:
      self.medfilt_kernel_size = params['medfilt_kernel_size']
      del params['medfilt_kernel_size']
    for node in self.nodes:
      self.models[node].set_params(**params)
    return self
