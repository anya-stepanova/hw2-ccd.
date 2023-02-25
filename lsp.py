import numpy as np

def lstsq_ne(a, b):
    x1 = np.linalg.inv(np.dot(a.T, a))
    x2 = np.dot(x1, a.T)
    x = np.dot(x2, b)
    
    bb = np.dot(a, x)
    
    cost = (b - bb)**2
    var = np.linalg.inv(np.dot(a.T, a))
    
    return (x, cost, var)

'''def lstsq_svd(a, b, rcond=None):
    v, s, u = np.linalg.svd(a)
    if rcond is None:
        where = (s != 0.0)
    else:
        where = s > s[0] * rcond
    f = u.T @ np.linalg.inv(s) @ v.T
    x = np.dot(f, b)
    bb = np.dot(a, x)
    
    cost = (b - bb)**2
    sigma = cost/(b.shape[0] - x.shape[0])
    var = np.dot(np.dot(u.T, np.diag(s**(-2))),u*sigma)
    
    return (x, cost, var)'''

def lstsq(a, b, method, **kwargs):
    if method == 'ne':
        return lstsq_ne(a, b)
    if method == 'svd':
        return lstsq_svd(a, b, **kwargs)