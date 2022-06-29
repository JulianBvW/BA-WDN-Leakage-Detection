import numpy as np
import pandas as pd

def past_days_transform(X, nodes=['10', '11', '12', '13', '21', '22', '23', '31', '32'], past_start=1, past_end=2):
  """Every point in the time series will include the data from specified prior 
  days at the exact hour ie. the pressure values for day 3 hour 5 will include 
  the pressure values of day 2 hour 5 if that day difference is wanted.

  Args:
    X (list of pd.DataFrame): The dataset.
    nodes (list of nodes): The nodes past data should be included.
    past_start (int): The first past day.
    past_end (int): The first past day not to be included.
  
  Returns:
    X: List of pd.Dataframe containing transformed node pressures.
  """

  # Raise exception if type is not list of pd.DataFrame
  if type(X) == np.ndarray:
    raise TypeError('X must be of type list of pd.DataFrame.')

  # Return if there would be no transformation
  if past_start >= past_end:
    return X

  # Transform every datapoint
  X_new = []
  for x in X:

    # For every day wanted
    for past_day in range(past_start, past_end):

      # The first days without real 'past' will just copy the first day
      fillers = [x.loc[:24-1, nodes].add_suffix(f'_past{past_day}') for i in range(past_day)]

      # Then add the shifted days
      shifted = [x.loc[:len(x)-24*past_day-1, nodes].add_suffix(f'_past{past_day}')]

      # Concatenate the results
      past = pd.concat(fillers + shifted)
      past.reset_index(drop=True, inplace=True)
      x = pd.concat([x, past], axis=1)[:len(x)]
    
    # Append to new list, because apperently 'for in' passes by value, not by reference
    X_new.append(x)
  
  return X_new

def mean_transform(X, window=3):
  """Creates a rolling window and calculates the mean value.

  Args:
    X (list of pd.DataFrame): The dataset.
    window (int): The rolling window size.

  Returns:
    X: List of pd.Dataframe containing transformed node pressures.
  """

  # Raise exception if type is not list of pd.DataFrame
  if type(X) == np.ndarray:
    raise TypeError('X must be of type list of pd.DataFrame.')

  # Return if there would be no transformation
  if window <= 1:
    return X
  
  # Use rolling window
  rolling = lambda X_single: X_single.rolling(window, min_periods=1).mean()
  
  return list(map(rolling, X))