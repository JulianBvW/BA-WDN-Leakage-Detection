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
  
  def get_nodes(self, nodes):
    """Select specific nodes

    Args:
      nodes (iterable of nodes): The nodes you want to get.
    
    Returns:
      Pandas Dataframe containing node pressures with hour as index.
    """
    return self.data.loc[:,nodes]