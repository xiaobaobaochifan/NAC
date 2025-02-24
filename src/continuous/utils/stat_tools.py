import numpy as np

def statistics_scalar(x, with_min_and_max=False):
    """
    Get mean, std, and optional min/max of scalar x using numpy operations.

    Args:
        x: An array containing samples of the scalar to produce statistics for.
        with_min_and_max (bool): If true, return min and max of x in addition 
                                 to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    mean = np.mean(x)  # Calculate mean using numpy
    std = np.std(x)    # Calculate standard deviation using numpy

    if with_min_and_max:
        min_x = np.min(x)  # Calculate minimum using numpy
        max_x = np.max(x)  # Calculate maximum using numpy
        return mean, std, min_x, max_x
    
    return mean, std