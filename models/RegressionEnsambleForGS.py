"""
This is an edit of the original RegressionEnsamble.py file
used for my own GridSearchCV."""

import numpy as np
import pandas as pd
from pexpect.spawnbase import _NullCoder
from scipy.signal import medfilt
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from utils.helper import any_transform
from sklearn.base import clone
from tqdm import tqdm

class RegressionEnsambleForGS():


  def __init__(self, model, nodes=['10','11','12','13','21','22','23','31','32']):

    self.models = {}
    for node in nodes:
      self.models[node] = clone(model)

    self.nodes = nodes


  def fit(self, X, y, verbose=False):

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

    self.thresholds_simple = abs(differences[y_concat == 0]).max()

    differences['hour of the day'] = X_concat['hour of the day']
    self.thresholds_daytime = abs(differences[y_concat == 0]).groupby(['hour of the day']).max()

    return self


  def predict_differences_list(self, X, verbose=False):

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
      differences['hour of the day'] = X_single['hour of the day']
      preds.append(differences)
    
    return preds

  def get_prediction_without_medfilt(self, differences_list, th_mode, th_multiplier, th_majority):
    min_voters = len(self.nodes) * th_majority
    
    preds = []
    for differences in differences_list:

      if th_mode == 'simple':
        all_without_daytime = list(differences.columns)
        all_without_daytime.remove('hour of the day')
        self.thresholds_simple *= th_multiplier
        pred = (differences[all_without_daytime] > self.thresholds_simple).sum(axis=1)
        pred = (pred >= min_voters).astype(int)
      else:
        self.thresholds_daytime *= th_multiplier
        by_daytimes = []
        for daytime in differences['hour of the day'].unique():
          by_daytime = differences[differences['hour of the day'] == daytime]
          by_daytimes.append(by_daytime > self.thresholds_daytime.loc[daytime])
        pred = pd.concat(by_daytimes).sort_index().sum(axis=1)
        pred = (pred >= min_voters).astype(int)
      
      preds.append(pred)

    return preds
