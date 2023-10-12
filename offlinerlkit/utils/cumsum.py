import numpy as np

def discount_cumsum(x: np.ndarray, gamma: float = 1.):
    '''
    Used to calculate rtg for rewards seq (x)
    '''
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum
