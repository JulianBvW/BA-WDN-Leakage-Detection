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
from utils.gs_utils import do_gridsearch
from utils.Network import WDN
from utils.Datagenerator import Datagenerator
from models.RegressionEnsambleForGS import RegressionEnsambleForGS

####### Main
LEAKDB_PATH = '../Net1_CMH/'

# New class for Neural Networks

class NNSize(object):
    def __init__(self, shape_min=1, shape_max=3, size_min=5, size_max=24):
        self.shape_min = shape_min
        self.shape_max = shape_max
        self.size_min = size_min
        self.size_max = size_max

    def rvs(self, random_state=None):
        #np.random.seed(random_state)
        shape = np.random.randint(self.shape_min, self.shape_max+1)
        return tuple([np.random.randint(self.size_min, self.size_max+1) for _ in range(shape)])

# Generate regression results

def rgen_regression(X, y, model='all'):
  print('### GENERATING REGRESSION RESULTS...')

  if model in ['all', 'lr']:
    rgen_regression_lr(X, y)
  if model in ['all', 'ridge']:
    rgen_regression_ridge(X, y)
  if model in ['all', 'lasso']:
    rgen_regression_lasso(X, y)
  if model in ['all', 'dt']:
    rgen_regression_dt(X, y)
  if model in ['all', 'rf']:
    rgen_regression_rf(X, y)
  if model in ['all', 'knn']:
    rgen_regression_knn(X, y)
  if model in ['all', 'svr']:
    rgen_regression_svr(X, y)
  if model in ['all', 'mlp']:
    rgen_regression_mlp(X, y)

  print('### RESULTS GENERATED...')

# Helper

def rgen_regression_lr(X, y, general_params):
  model_lr = LinearRegression
  parameters_lr = {}
  results, matric_keys = do_gridsearch(X, y, model_lr, parameters_lr)
  rs_mean = round(results.groupby(matric_keys).mean(), 3)
  results.to_csv(f'results/regression_lr_full.csv')
  rs_mean.to_csv(f'results/regression_lr_mean.csv')

def rgen_regression_ridge(X, y, general_params):
  model_ridge = Ridge
  parameters_ridge = {'alpha': [0.333*i for i in range(1, 7)]}
  results, matric_keys = do_gridsearch(X, y, model_ridge, parameters_ridge)
  rs_mean = round(results.groupby(matric_keys).mean(), 3)
  results.to_csv(f'results/regression_ridge_full.csv')
  rs_mean.to_csv(f'results/regression_ridge_mean.csv')

def rgen_regression_lasso(X, y, general_params):
  model_lasso = Lasso
  parameters_lasso = {'alpha': [0.333*i for i in range(1, 7)], 'max_iter': [2000]}
  results, matric_keys = do_gridsearch(X, y, model_lasso, parameters_lasso)
  rs_mean = round(results.groupby(matric_keys).mean(), 3)
  results.to_csv(f'results/regression_lasso_full.csv')
  rs_mean.to_csv(f'results/regression_lasso_mean.csv')

def rgen_regression_dt(X, y, general_params):
  model_dt = DecisionTreeRegressor
  parameters_dt = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                   'splitter': ['best', 'random']}
  results, matric_keys = do_gridsearch(X, y, model_dt, parameters_dt)
  rs_mean = round(results.groupby(matric_keys).mean(), 3)
  results.to_csv(f'results/regression_dt_full.csv')
  rs_mean.to_csv(f'results/regression_dt_mean.csv')

def rgen_regression_rf(X, y, general_params):
  model_rf = RandomForestRegressor
  parameters_rf = {'criterion': ['squared_error', 'absolute_error']}
  results, matric_keys = do_gridsearch(X, y, model_rf, parameters_rf)
  rs_mean = round(results.groupby(matric_keys).mean(), 3)
  results.to_csv(f'results/regression_rf_full.csv')
  rs_mean.to_csv(f'results/regression_rf_mean.csv')

def rgen_regression_knn(X, y, general_params):
  model_knn = KNeighborsRegressor
  parameters_knn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
  results, matric_keys = do_gridsearch(X, y, model_knn, parameters_knn)
  rs_mean = round(results.groupby(matric_keys).mean(), 3)
  results.to_csv(f'results/regression_knn_full.csv')
  rs_mean.to_csv(f'results/regression_knn_mean.csv')

def rgen_regression_svr(X, y, general_params):
  model_svr = SVR
  parameters_svr = {'kernel': ['linear', 'poly', 'rbf'], 
                    'C': [10**e for e in range(-1, 2)], 
                    'gamma': [0.001, 0.002, 0.003]}
  results, matric_keys = do_gridsearch(X, y, model_svr, parameters_svr)
  rs_mean = round(results.groupby(matric_keys).mean(), 3)
  results.to_csv(f'results/regression_svr_full.csv')
  rs_mean.to_csv(f'results/regression_svr_mean.csv')

def rgen_regression_mlp(X, y, general_params):
  model_mlp = MLPRegressor
  parameters_mlp = {'hidden_layer_sizes': [NNSize().rvs() for _ in range(30)], 
                    'learning_rate': ['constant', 'adaptive'], 
                    'activation': ['logistic', 'tanh', 'relu']}
  results, matric_keys = do_gridsearch(X, y, model_mlp, parameters_mlp)
  rs_mean = round(results.groupby(matric_keys).mean(), 3)
  results.to_csv(f'results/regression_mlp_full.csv')
  rs_mean.to_csv(f'results/regression_mlp_mean.csv')

# Main

def main():
  wdn = WDN("nets/Net1.inp", ['10', '11','12','13','21','22','23','31','32'])
  gen = Datagenerator(wdn)

  print('### LOADING DATASET...')
  X, y = gen.get_dataset(LEAKDB_PATH, size=500)

  if len(sys.argv) > 1:
    rgen_regression(X, y, model=sys.argv[1])
  else:
    rgen_regression(X, y)

if __name__ == '__main__':
  main()
