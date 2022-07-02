""" This is for my own GridSearchCV implementation
specifically for this task."""

import random
import pandas as pd
from tqdm import tqdm
from scipy.signal import medfilt
from sklearn.model_selection import ParameterGrid

from models.RegressionEnsamble import RegressionEnsamble
from models.RegressionEnsambleForGS import RegressionEnsambleForGS
from utils.metrics import accuracy, recall, specificity, precision, detection_time_mean, detection_time_std, detection_time_median
from utils.feature_extraction import past_days_transform, mean_transform


# Default parameter lists
param_grid_regr = {'th_mode': ['simple', 'daytime'], 'th_multiplier': [1.0 + 0.05*i for i in range(5)], 'th_majority': [i / 10 for i in range(1, 11)]}
param_grid_mk = {'mk_size': [3+2*i for i in range(3)]}


def select(l, idxs):
  """Implements 'Multiple Select' for Lists.
     e.g. my_list[[2,7,100]]"""
  l_selected = []
  for idx in idxs:
    l_selected.append(l[idx])
  return l_selected

def kfold(cv=5, from_i=0, to_i=250):
  """Get Train and Test indices."""
  idxs = list(range(from_i, to_i))
  random.shuffle(idxs)
  size = (to_i - from_i) // cv
  start = 0
  for _ in range(cv):
    #yield      train                           test
    yield idxs[:start] + idxs[start+size:], idxs[start:start+size]
    start += size

def cv_split(X, y, perc=0.5, cv=5):
  """Split the Dataset using kfold()."""
  size = len(X)
  middle = int(size*perc)
  for (f_y1_train, f_y1_test), (f_y0_train, f_y0_test) in zip(kfold(cv=cv, to_i=middle), kfold(cv=cv, from_i=middle, to_i=size)):
    X_train, X_test = select(X, f_y1_train+f_y0_train), select(X, f_y1_test+f_y0_test)
    y_train, y_test = select(y, f_y1_train+f_y0_train), select(y, f_y1_test+f_y0_test)
    yield X_train, X_test, y_train, y_test

def get_results(y_true_list, y_pred_list):
  """Creates a metric dictionary."""
  return {
      'accuracy': accuracy(y_true_list, y_pred_list),
      'recall': recall(y_true_list, y_pred_list),
      'specificity': specificity(y_true_list, y_pred_list),
      'precision': precision(y_true_list, y_pred_list),
      'dt_mean': detection_time_mean(y_true_list, y_pred_list),
      'dt_std': detection_time_std(y_true_list, y_pred_list),
      'dt_median': detection_time_median(y_true_list, y_pred_list)
  }

def do_gridsearch(X, y, base_model, param_grid_model, cv=5):
  results = []

  # Do Cross Validation
  for cv_step, (X_train, X_test, y_train, y_test) in enumerate(cv_split(X, y, cv=cv)):
    print('# Start of CV step', cv_step)

    # Train the base model
    for params_model in tqdm(ParameterGrid(param_grid_model)):
      model = RegressionEnsambleForGS(base_model(**params_model))
      model.fit(X_train, y_train)
      diff_preds = model.predict_differences_list(X_test)

      # Postprocess the predicted differences
      for params_regr in tqdm(ParameterGrid(param_grid_regr), leave=False):#ParameterGrid(param_grid_regr):
        preds = model.get_prediction_without_medfilt(diff_preds, **params_regr)

        # Apply Median Filter
        for params_mk in ParameterGrid(param_grid_mk):
          preds_medfilt = [medfilt(pred, params_mk['mk_size']) for pred in preds]
          results.append(dict({'cv_step': cv_step}, **params_model, **params_regr, **params_mk, **get_results(y_test, preds_medfilt)))

  results_df = round(pd.DataFrame(results), 3)
  parameter_keys = [*param_grid_model] + [*param_grid_regr] + [*param_grid_mk]
  return results_df, parameter_keys

def do_fe_gridsearch(X, y, base_model, param_grid_fe, params_ensamble, cv=5):
  results = []

  for params_fe in tqdm(ParameterGrid(param_grid_fe)):

    # Apply transformations
    X_transformed = mean_transform(X, window=params_fe['window'])
    X_transformed = past_days_transform(X_transformed, past_end=params_fe['past_end'])

    # Do Cross Validation
    for cv_step, (X_train, X_test, y_train, y_test) in enumerate(cv_split(X_transformed, y, cv=cv)):
      print('# Start of CV step', cv_step)

      # Train the model
      model = RegressionEnsamble(base_model, **params_ensamble)
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      results.append(dict({'cv_step': cv_step}, **params_fe, **get_results(y_test, y_pred)))

  results_df = round(pd.DataFrame(results), 3)
  parameter_keys = [*param_grid_fe]
  return results_df, parameter_keys
