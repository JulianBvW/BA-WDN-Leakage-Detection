import numpy as np
import pandas as pd

class Dataloader:
  """
  Class used to reshape WDN data.
  """

  def __init__(self, simulation_results, important_nodes):
    """Create a dataset based on the simulated results.

    Args:
      simulation_results (Dataframe): Simulated results including hour, time of day, day nad pressures.
      importnat_nodes (iterable of nodes): List of nodes (Strings) interesting for ML and plotting.
    """
    self.data = simulation_results
    self.important_nodes = important_nodes
  
  def get_nodes(self, nodes=[], include_time=False):
    """Select specific nodes or just the important nodes.

    Args:
      nodes (iterable of nodes) (optional): The nodes you want to get.
    
    Returns:
      Pandas Dataframe containing node pressures with hour as index.
    """
    nodelist = nodes

    if not nodes:
      nodelist += self.important_nodes
    
    if include_time:
      nodelist += ['hour of the day', 'day']
      
    return self.data.loc[:, nodelist]
  
  def get_days_at_hour(self, hour, nodes=[]):
    """Set day as index while looking at a specific hour.

    Args:
      hour (Int): The hour of the day.
      nodes (iterable of nodes) (optional): The nodes you want to get.
    
    Returns:
      pandas Dataframe containing pressure values for the specified hour.
    """
    at_hour = self.data[self.data['hour of the day'] == hour].set_index('day')
    
    if nodes:
      return at_hour.loc[:, nodes]

    return at_hour.loc[:, self.important_nodes]
