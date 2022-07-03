# General
import sys
import numpy as np
import pandas as pd

# SKLearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Own
from utils.gs_utils import do_fe_gridsearch
from utils.Network import WDN
from utils.Datagenerator import Datagenerator

####### Main
LEAKDB_PATH = '../Net1_CMH/'

# Generate feature extraction results

def rgen_fe(X, y, base_model, params_ensamble):
  param_grid_fe = {'window': range(1, 6), 'past_end': range(1, 6)}
  results, metric_keys = do_fe_gridsearch(X, y, base_model, param_grid_fe, params_ensamble)
  rs_mean = round(results.groupby(metric_keys).mean(), 3)
  results.to_csv(f'results/regression_fe_mlp_full.csv')
  rs_mean.to_csv(f'results/regression_fe_mlp_mean.csv')

# Main

def main():
  print('### LOADING NETWORK')
  wdn = WDN("nets/Net1.inp", ['10', '11','12','13','21','22','23','31','32'])
  gen = Datagenerator(wdn)

  print('### LOADING DATASET...')
  X, y = gen.get_dataset(LEAKDB_PATH, size=500)

  params_ensamble = {'th_mode': 'daytime', 'th_multiplier': 1.00, 'th_majority': 0.1, 'mk_size': 3}
  rgen_fe(X, y, MLPRegressor(hidden_layer_sizes=(7, 6), learning_rate='adaptive', activation='relu'), params_ensamble)

if __name__ == '__main__':
  main()
