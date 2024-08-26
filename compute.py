'''
Compute a sharp (best-possible) upper bound on the population (or sample) variance statistic. 

This code works independently of interval arithmetic libraries. 

Only Numpy is needed for this code to work. Scipy is needed for the quadratic algorithm only, i.e. to enable comparisons.

The data needed to compute the population variance must have shape (n,2), where 2 is the imprecision dimension.

Dependencies

* Numpy
* Scipy (optional, for quadratic algorithm) 
* Matplotlib Pyplot (optional, for plotting)


'''

import numpy as np
from numpy import (ndarray,asarray)
from scipy.optimize import (Bounds, minimize) # , LinearConstraint
from typing import Callable


def a(x,dtype=float): return asarray(x,dtype=dtype)

# Data must have shape (n,2), where 2 is the lower and upper bound.
# This code is standalone in the sense that need not run interval arithemtic under the hood.

COUNTER_VAR = []
COUNTER_GRAD = []
COUNTER_HESS = []

###############################################################################################################################################
############################################  VARIANCE STATISTIC ##############################################################################
###############################################################################################################################################
def pvar(x:ndarray): return np.var(x)                   # Numpy's population (or unbiased sample) variance
def svar(x:ndarray): return 1/(1-1/len(x)) * pvar(x)    # sample variance from Numpy's variance
def svar_(x:ndarray,population:Callable): return 1/(1-1/len(x)) * population(x) # sample variance from any population variance
def var(x:ndarray, corrected=False): # computes the population variance if unbiased=True, computes the unbiased sample variance otherwise
    if corrected: return svar(x)
    else: return pvar(x)
def wvar(x:ndarray,w:ndarray=None): # computes the weighted population variance with Numpy algorithms. Supposedly without cancellation errors.
    '''
    x and w have shape (n,).
    '''
    COUNTER_VAR.append(1) # Used to count funtion evaluations. Comment this line out for speed.
    if x.__class__.__name__!='ndarray': x = a(x,dtype=float)
    if len(x.shape)>1: raise ValueError(f'x must have shape (n,) while shape given is {x.shape}.')
    if w is None: 
        n=len(x)
        w = a(n*[1/n])
    elif w.__class__.__name__!='ndarray': w = a(w,dtype=float)
    mu = np.average(x,weights=w)
    d2 = (x-mu)**2
    return np.average(d2,weights=w) 
def gradient_variance_x(x:ndarray): # Gradient of the variance expression w.r.t. x.
    '''
    x has shape (n,).

    x: interval data
    output: d/dx f(x)
    '''
    if x.__class__.__name__!='ndarray': x = a(x,dtype=float)
    if len(x.shape)>1: raise ValueError(f'x must have shape (n,) while shape given is {x.shape}.')
    n = len(x)
    mu = np.sum(x)/n
    return (2/n)*(x-mu)
def hessian_variance_x(x:ndarray): # Hessian of variance expression w.r.t. x
    '''
    x has shape (n,).

    x: interval data. 
    output: d2/dx2 f(x)
    '''
    n = len(x)
    a = (1-n) * np.eye(n) + (np.ones((n,n)) - np.eye(n))
    b = -2/n**2 * np.ones((n,n))
    return a*b
##############################################################################################################################################
##############################################################################################################################################


############################################################################################################
############################################ VARIANCE OF MIXTURES ##########################################
############################################################################################################
def parse_inputs(x:ndarray,p:ndarray,m:ndarray=None):
    """
    x has shape (n,2), p and m have shape (n,)
    """
    if x.__class__.__name__!='ndarray': x = a(x,dtype=float)
    if p.__class__.__name__!='ndarray': p = a(p,dtype=float)
    if len(x.shape)==1: raise ValueError(f'x must have shape (n,2) while shape given is {x.shape}.')
    if len(x.shape)>2: raise ValueError(f'x must have shape (n,2) while shape given is {x.shape}.')
    n,_ = x.shape
    if m is None: m = a(n*[1/n]) # equal-weight focal elements
    elif m.__class__.__name__!='ndarray': m = a(m,dtype=float) 
    return x,p,m
def dispersive_variance(x:ndarray,p:ndarray,m:ndarray=None): # Variance of two-discrete mixture distribution
    '''
    x has shape (n,2) and p, m have shape (n,). 

    x: interval data. 
    p: two-discrete distribution left endpoint only, 0 <= p <= 1.
    m: mass of the focal element, if None m = 1/n.

    output: v(p)
    '''
    x,p,m = parse_inputs(x,p,m)
    p2 = np.vstack((p,1-p)).T # p2.shape = (n,2)
    x__=x.flatten(order='C')
    pm = (p2.T*m).T # transpose game to enable broadcasting
    p__=pm.flatten(order='C')
    return wvar(x__,p__)
def gradient_dispersive_variance_p(x:ndarray,p:ndarray,m:ndarray=None):
    '''
    x has shape (n,2), p and m have shape (n,). 

    x: interval data. 
    p: two-discrete distribution left endpoint only, 0 <= p <= 1.
    m: mass of the focal element, if None, m = 1/n.
    '''
    COUNTER_GRAD.append(1)
    x,p,m = parse_inputs(x,p,m)
    w = x[:,1]-x[:,0]       # interval width
    c = (x[:,0]+x[:,1])/2   # interval midpoint
    mu_hi = np.sum(m*x[:,1])
    return -2*m*w*(c+np.sum(m*p*w)-mu_hi)
def hessian_dispersive_variance_p(x:ndarray,p:ndarray,m:ndarray=None):
    '''
    `x` has shape `(n,2)`, `p` and `m` have shape `(n,)`. 

    x: interval data. 
    p: two-discrete distribution left endpoint only, `0 <= p <= 1`.
    m: mass of the focal element, if None, `m = 1/n`.
    '''
    COUNTER_HESS.append(1)
    x,p,m = parse_inputs(x,p,m)
    w = x[:,1]-x[:,0] # interval width
    mw = m*w
    return -2*np.outer(mw,mw)
############################################################################################################
############################################################################################################


############################################################################################################
############################################  EXACT ALGORITHM O(2n) ########################################
############################################################################################################
def bi_to_mask(bi:np.ndarray,dim=2):
    ''' Turn a vector of 0s and 1s e.g. (1,0,0,0,1) into a masking array [(0,1),(1,0),(1,0),(1,0),(0,1)].
    If dim > 2,  e.g. (2,0,1,0) the mask is [(0,0,1),(1,0,0),(0,1,0),(1,0,0)]. '''
    bi = np.asarray(bi,dtype=int)
    return np.asarray([bi==j for j in range(dim)],dtype=bool).T

def exact_algorithm_2n(x:ndarray,corrected=False): # Computes variance at every corner of the space product.
    '''
    x has shape (n,2). 
    '''
    if x.__class__.__name__!='ndarray': x = a(x,dtype=float)
    if len(x.shape)==1: raise ValueError(f'x must have shape (n,2) while shape given is {x.shape}.')
    if len(x.shape)>2: raise ValueError(f'x must have shape (n,2) while shape given is {x.shape}.')
    n,two = x.shape
    if two!=2: raise ValueError(f'x must have shape (n,2) while shape given is {x.shape}.')
    n=x.shape[0]
    if n > 21: 
        print('The exact algorithm was interrupted because the lenght of data exceeded capacity (n > 21).')
        return 0,0
    candidate_val=-1 # only for positive functions
    for j in range(2**n):
        b = tuple([j//2**h-(j//2**(h+1))*2 for h in range(n)]) # tuple of 0s and 1s
        s = bi_to_mask(b)
        new_val = wvar(x[s])# new_val = var(x[s],corrected=corrected) # scalar
        if new_val > candidate_val:
            candidate_val =  new_val
            candidate_corner = b
    exact_max = candidate_val
    exact_corner = candidate_corner
    f_count = sum(COUNTER_VAR)
    return exact_max, exact_corner, f_count
############################################################################################################
############################################################################################################


#####################################################################################################################################################
##########################################################  QUADRATIC ALGORITHM  ####################################################################
#####################################################################################################################################################
def quadratic_algorithm(x:ndarray, m:ndarray=None, precision=0.001, corrected=False, verbose=1):# Computes the sharp upper bound via quadratic optimization.
    '''x has shape (n,2). m has shape (n,), the mass m is assigned to each focal element.'''
    def adjust_for_precision(p_,precision=0.001):
        zero_ = p_ < precision
        one_  = (1-p_) < precision
        p_[zero_] = 0 
        p_[one_]  = 1
        return p_
    def realvalued_argmax(p_): return a(p_<.5,dtype=int)
    def objective(p): return -dispersive_variance(x,p,m=m)
    def jac(p): return -gradient_dispersive_variance_p(x,p,m=m)
    def hess(p): return -hessian_dispersive_variance_p(x,p,m=m)
    x,_,m = parse_inputs(x,0,m)
    n,two = x.shape
    if two!=2: raise ValueError(f'Shape of x must be (n,2), shape provided is {x.shape}.')
    bounds = Bounds([0]*n,[1]*n) # https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
    p0 = a([.5]*n) # initial distribution
    res = minimize(objective, p0, method='trust-constr', jac=jac, hess=hess, options={'verbose': verbose}, bounds=bounds)
    pq_max,res_f = res.x, res.fun
    pq_adj = adjust_for_precision(pq_max,precision=precision) # only left endpoints
    v_ub = dispersive_variance(x,pq_max,m)
    v_adj = dispersive_variance(x,pq_adj,m)
    if v_adj>v_ub: # Adjusting value of p to force integer solutions when sufficiently close to {0,1}
        v_ub = v_adj
        pq_max = pq_adj
    else: print('Adjusting to enforce integer solution did not work.')
    corners_x = realvalued_argmax(pq_max) # 0 <- lo, 1 <- hi
    real_var = var(x[bi_to_mask(corners_x)],corrected=corrected)
    v_count = sum(COUNTER_VAR)
    g_count = sum(COUNTER_GRAD)
    h_count = sum(COUNTER_HESS)
    return v_ub, pq_max, real_var, corners_x, v_count, g_count, h_count
#####################################################################################################################################################
#####################################################################################################################################################


#####################################################################################################################################
############################################  POLYNOMIAL ALGORITHM  #################################################################
#####################################################################################################################################
def Threshold(x_:ndarray,p_:ndarray,m_:ndarray=None):
    """
    x_.shape : (n,2), p_. m_.shape : (n,).
    """
    x_,p_,m_ = parse_inputs(x_,p_,m_)
    _,two = x_.shape
    if two!=2: raise ValueError(f'Shape of x must be (n,2), shape provided is {x_.shape}.')
    x_lo,x_hi = x_[:,0],x_[:,1]
    mu_hi = np.sum(m_*x_hi)
    wids = x_hi-x_lo
    return mu_hi - np.sum(m_*p_*wids)

def Centre(x_:ndarray,m_:ndarray=None,h_:ndarray[bool]=None):
    """
    x_.shape : (n,2), m_.shape : (n,).
    """
    x_,_,m_ = parse_inputs(x_,0,m_)
    _,two = x_.shape
    if two!=2: raise ValueError(f'Shape of x must be (n,2), shape provided is {x_.shape}.')
    x_lo,x_hi = x_[:,0],x_[:,1]
    wids = x_hi - x_lo
    mids = (x_hi + x_lo)/2
    if h_ is None: mw = np.sum(wids*mids)/np.sum(wids) 
    else: mw = np.sum(wids[h_]*mids[h_])/np.sum(wids[h_]) 
    p_0 = (mids < mw)*1
    mu_hi = np.sum(m_*x_hi)
    t_0 = mu_hi - np.sum(m_*p_0*wids)
    return mw, p_0, t_0

def Mean(x_:ndarray,m_:ndarray=None): 
    """
    x_.shape : (n,2), m_.shape : (n,).
    """
    x_,_,m_ = parse_inputs(x_,0,m_)
    _,two = x_.shape
    if two!=2: raise ValueError(f'Shape of x must be (n,2), shape provided is {x_.shape}.')
    x_lo,x_hi = x_[:,0],x_[:,1]
    return np.mean(x_lo), np.mean(x_hi)

def threshold_task(x:ndarray,m=None,corrected=False):
    '''
    x.shape : (n,2), m.shape : (n,).
    '''
    def realvalued_argmax(p_:ndarray): return a(p_<.5,dtype=int)
    ############################################ Interval methods ####################################################################################
    def assign_threshold_interval(m_:float,t_:float): 
        if t_<=m_: return a([t_,m_],dtype=float)
        if m_< t_: return a([m_,t_],dtype=float)
    def all_self_equal(x_:ndarray)->bool: return np.all(np.isclose(0,np.diff(x_),atol=0,rtol=1e-09)) # True is al elements in x_ are equal.
    def contain(x_:ndarray,y_:ndarray): # from intervals library but without interval class
        if (len(x_.shape) >1) & (len(y_.shape) >1): return (x_[:,0]<=y_[:,0]) & (x_[:,1]>=y_[:,1]) # x contain y
        if (len(x_.shape)==1) & (len(y_.shape) >1): return (x_[0]  <=y_[:,0]) & (x_[1]  >=y_[:,1]) # x contain y
        if (len(x_.shape) >1) & (len(y_.shape)==1): return (x_[:,0]<=y_[0])   & (x_[:,1]>=y_[1]) # x contain y
        if (len(x_.shape)==1) & (len(y_.shape)==1): return (x_[0]  <=y_[0])   & (x_[1]  >=y_[1]) # x contain y
    def intersection(x_:ndarray,y_:ndarray): # from intervals library but without interval class
        '''
        x and y have shape (2,), i.e. they are just one interval
        '''
        x_lo,x_hi,y_lo,y_hi = x_[0],x_[1],y_[0],y_[1]
        if (x_hi<y_lo) or (y_hi<x_lo): return np.nan  # should return an empty interval
        if contain(x_,y_): return y_ # x contain y
        if contain(y_,x_): return x_ # y contain x
        if (x_lo < y_lo) and (y_lo < x_hi): return a([y_lo,x_hi])
        if (y_lo < x_lo) and (x_lo < y_hi): return a([x_lo,y_hi])
    ############################################ ################# ####################################################################################
    x,_,m = parse_inputs(x,0,m)
    n,two = x.shape
    nn = a(range(n),dtype=int)
    if two!=2: raise ValueError(f'Shape of x must be (n,2), shape provided is {x.shape}.')
    x_lo,x_hi = x[:,0],x[:,1]
    mu_hi = np.sum(m*x_hi)
    wids = x_hi-x_lo
    mids = (x_hi + x_lo)/2
    def center_of_rotation(h_:ndarray[bool]=None):
        '''
        h_.shape : (n,)
        '''
        if h_ is None: return np.sum(wids*mids)/np.sum(wids) 
        else: return np.sum(wids[h_]*mids[h_])/np.sum(wids[h_]) 
    def threshold(p_): return mu_hi - np.sum(m*p_*wids)
    mw_0 = center_of_rotation()           # **** weighted average of all midpoints *****
    p_0 = (mids < mw_0)*1                 # quasi-optimal distribution.
    t_0 = threshold(p_0)                  # t_wm != wm
    threshold_interval = assign_threshold_interval(mw_0,t_0)
    h_mask=a(contain(threshold_interval,a(2*[mids]).T),dtype=bool)
    H_cardinality=np.sum(h_mask)
    print(f'First threshold interval contains {H_cardinality} midpoint(s).')
    ################################################################################################
    def closed_form_solution(p_:ndarray,h_:ndarray[bool]):
        '''
        p_ and h_ have shape (n,)
        '''
        p_cf=np.asarray(p_.copy(),dtype=float)
        p_cf[h_] = n*(mu_hi - mids[h_] - sum(wids[~h_]*p_cf[~h_])/n)/wids[h_] # closed-form expression for #H=1
        t_cf = threshold(p_cf)
        mw_cf = mids[h_mask][0]
        tI_cf = assign_threshold_interval(mw_cf,t_cf)
        return p_cf,t_cf,mw_cf,tI_cf
    def prepare_for_gradient_ascent(h_:ndarray[bool]):
        '''
        h_.shape : (n,)
        '''
        mw_hh = mids[h_][0] # all equals so take the first one.
        p_hh = (mids < mw_hh)*1.
        p_hh[h_]=.5 # non-integer solutions must be determined via gradient ascent. Here they are initialised to 0.5
        t_hh = threshold(p_hh) 
        tI_hh = assign_threshold_interval(mw_hh,t_hh)
        return p_hh,t_hh,mw_hh,tI_hh
    def threshold_iteration(h_:ndarray[bool],tI_0:ndarray):
        '''
        h_.shape : (n,)
        tI_0.shape : (2,)
        '''
        mw_kk = center_of_rotation(h_)
        p_kk = (mids < mw_kk)*1.                                         # quasi-optimal distribution.
        t_kk = threshold(p_kk)                                          # t_wm != wm
        new_tI = assign_threshold_interval(mw_kk,t_kk)
        new_tI = intersection(new_tI,tI_0)
        tI_k = new_tI+0.
        return p_kk,t_kk,mw_kk,tI_k
    ################################################################################################
    if H_cardinality==0: # Exact upper bound.
        print('Exact bound was found.')
        p_out = p_0
        t_out = t_0
    elif H_cardinality==1: # closed-form solution -- upper bound > maximum (not exact)
        p_cf,t_cf,mw_cf,tI_cf = closed_form_solution(p_0,h_mask)
        print(f'Solution found in closed form becuase #H={H_cardinality}. Gradient task not needed.')
        print(f'Closed form solution = {p_cf[h_mask]} for element # {nn[h_mask]}')
        p_out = p_cf
        t_out = t_cf
    elif H_cardinality>1:
        if all_self_equal(mids[h_mask]): # mid_h = mids[h_mask]
            print(f'There are {H_cardinality} midpoints of equal value. The gradient task is needed.')
            p_h,t_h,mw_h,tI_h = prepare_for_gradient_ascent(h_mask)
            p_out = p_h
            t_out = t_h
        else: # midpoints in range are not all equal
            print('Proceed with threshold iterations..')
            k=1
            tI_k = threshold_interval
            while True: # update h_mask
                p_k,t_k,mw_k,tI_k = threshold_iteration(h_mask,tI_k)
                h_mask = a(contain(tI_k,a(2*[mids]).T),dtype=bool)
                H_cardinality = np.sum(h_mask)
                if H_cardinality==0: # Exact upper bound.
                    p_out = p_k                
                    t_out = t_k   
                    print('Exact bound was found.')
                    break #for loop
                if H_cardinality==1: 
                    p_cf,t_cf,mw_cf,tI_cf = closed_form_solution(p_k,h_mask)
                    print(f'Solution found in closed form at threshold iteration {k}. No need for gradient task.')
                    print(f'Closed form solution = {p_cf[h_mask]} for element # {nn[h_mask]}')
                    p_out = p_cf
                    t_out = t_cf
                    break # for loop
                if all_self_equal(mids[h_mask]):
                    print(f'There are {H_cardinality} of equal value after {k} threshold iterations. The algorithm can move forward to gradient task.')
                    p_h,t_h,mw_h,tI_h = prepare_for_gradient_ascent(h_mask)
                    p_out = p_h
                    t_out = t_h
                    break # for loop 
                k+=1
    corners_x = realvalued_argmax(p_out) # 0 <- lo, 1 <- hi
    v = dispersive_variance(x,p_out,m=m)
    realvalued_var = var(x[bi_to_mask(corners_x)],corrected=corrected)
    v_count = sum(COUNTER_VAR)
    g_count = sum(COUNTER_GRAD)
    h_count = sum(COUNTER_HESS)
    return v, p_out, realvalued_var, corners_x, h_mask, t_out, v_count, g_count, h_count

def gradient_task(x:ndarray,p:ndarray,h:ndarray[bool],m=None,stop=1_000,e=1e-7,corrected=False):
    """
    x.shape : (n,2) 
    p.shape : (n,) 
    h.shape : (n,)
    """
    def realvalued_argmax(p_:ndarray): return a(p_<.5,dtype=int)
    x,p,m = parse_inputs(x,p,m)
    n,two = x.shape
    if two!=2: raise ValueError(f'Shape of x must be (n,2), shape provided is {x.shape}.')
    x_lo,x_hi = x[:,0],x[:,1]
    wids = x_hi-x_lo
    mu_hi = np.sum(m*x_hi)
    def threshold(p_:ndarray): return mu_hi - np.sum(m*p_*wids)
    if np.sum(h)<2: 
        print('Gradient task not needed.')
        corners_x = realvalued_argmax(p) # 0 <- lo, 1 <- hi
        real_var = var(x[bi_to_mask(corners_x)],corrected=corrected)
        v = dispersive_variance(x,p,m=m)
        t = threshold(p)
        v_count = sum(COUNTER_VAR)
        g_count = sum(COUNTER_GRAD)
        h_count = sum(COUNTER_HESS)
        return v, p, real_var, corners_x, 0, t, v_count, g_count, h_count
    if np.sum(h)>1: 
        mids = (x_hi + x_lo)/2
        def gradient(p_) : return gradient_dispersive_variance_p(x,p_,m=m)
        def gradient_step_nolr(p_:ndarray,g_:ndarray,mask:ndarray[bool]=None):
            new_p = p_.copy() *1.0 # must be float (no int!)
            g_normized = g_/np.linalg.norm(g_)
            p_max_step = p_.copy() *1.0 # must be float (no int!)
            p_max_step[g_>0] = 1. - new_p[g_>0] # p_max_step[g_<0] = new_p[g_<0]
            new_p[mask] = new_p[mask] + g_normized[mask]*p_max_step[mask] # gradient ascent step
            distmx = max(abs(new_p[mask]-p_[mask]))
            return new_p, distmx
        print(f'Gradient ascent begins.. ')
        for i in range(stop):
            g = gradient(p)                                   # compute the gradient. 
            p,dbp = gradient_step_nolr(p,g,h)             # learn-rate free.
            t = threshold(p)                          # gradient threshold. 
            dtcm = min(abs(t - mids[h])) # distance to closest midpoint
            if (dtcm<e) and (dbp<e):
                print(f'Gradient ascent stopped reaching the desired accuracy, after K={i+1}.')
                break # for loop
            if i==stop-1: print(f'optimum not found after {i+1} iterations.') # k_=i+1
    else: i=0
    t = threshold(p)
    corners_x = realvalued_argmax(p) # 0 <- lo, 1 <- hi
    real_var = var(x[bi_to_mask(corners_x)],corrected=corrected)
    v = dispersive_variance(x,p,m=m)
    v_count = sum(COUNTER_VAR)
    g_count = sum(COUNTER_GRAD)
    h_count = sum(COUNTER_HESS)
    return v, p, real_var, corners_x, i+1, t, v_count, g_count, h_count

def fast_algorithm(x:ndarray,m=None,corrected=False,stop=1_000,e=1e-7):
    '''
    x.shape : (n,2), m.shape : (n,).
    '''
    v, p, real_var, corners_x, h_mask, t, f_count, g_count, h_count = threshold_task(x,m,corrected=corrected)
    if sum(h_mask)>1: # start gradient ascent
        v, p, real_var, corners_x, k, t, f_count, g_count, h_count = gradient_task(x,p,h_mask,m=m,stop=stop,e=e,corrected=corrected)
        return v, p, real_var, corners_x, h_mask, k, t, f_count, g_count, h_count
    else: # threshold task is enough
        return v, p, real_var, corners_x, h_mask, 0, t, f_count, g_count, h_count
#####################################################################################################################################
#####################################################################################################################################

def dispargmax(c_:ndarray): 
    c = a(c_)
    n = len(c)
    s=a(n*['l'],dtype=str)
    s[c==1]='r'
    return s
    

def full_report(x,v,p,rv,corners,h,k,t,fcount,gcount,hcount):
    mw,p0,t0 = Centre(x)
    mu_lo,mu_hi = Mean(x)
    mids = (x[:,1] + x[:,0])/2
    print('------------------------------------------------------------------')
    print('---------------------- POLYNOMIAL ALGORITHM ----------------------')
    print('------------------------------------------------------------------')
    print(f'Midponits: {mids}')
    print(f'Interval mean: [{mu_lo}, {mu_hi}]')
    print(f'Width weighted mean: {mw}')
    print(f'Initial threshold: {t0}')
    print(f'Initial distribution: {p0}')
    print('------------------------------------------------------------------')
    print(f'Number of function evaluations v(p): {fcount}')
    print(f'Number of gradient evaluations g(p): {gcount}')
    print(f'Number of hessian evaluations h(p): {hcount}')
    print('------------------------------------------------------------------')
    print(f'Variance at optimum:  {v}')
    print(f'Optimum distribution: {p}')
    print(f'Threshold at optimum: {t}')
    cH = sum(h)
    print(f'Cardinality of H set: {cH}')
    if cH==0: print(f'Exact upper bound. Dispersive variance must coincide with real valued variance.')
    print(f'Number of gradient iterations: {k}')
    print('------------------------------------------------------------------')
    print(f'Real valued variance: {rv}')
    print(f'Corners: {dispargmax(corners)}')
    print('------------------------------------------------------------------')

def full_report_quadratic(v,p,rv,corners,fcount,gcount,hcount):
    print('------------------------------------------------------------------')
    print('---------------------- QUADRATIC ALGORITHM -----------------------')
    print('------------------------------------------------------------------')
    print(f'Number of function evaluations v(p): {fcount}')
    print(f'Number of gradient evaluations g(p): {gcount}')
    print(f'Number of hessian evaluations h(p): {hcount}')
    print('------------------------------------------------------------------')
    print(f'Variance at optimum:  {v}')
    print(f'Optimum distribution: {p}')
    print('------------------------------------------------------------------')
    print(f'Real valued variance: {rv}')
    print(f'Corners: {dispargmax(corners)}')
    print('------------------------------------------------------------------')

def full_report_exact(rv,corners,fcount):
    print('------------------------------------------------------------------')
    print('------------------------- EXACT ALGORITHM ------------------------')
    print('------------------------------------------------------------------')
    print(f'Number of function evaluations v(p): {fcount}')
    print('------------------------------------------------------------------')
    print(f'Real valued variance: {rv}')
    print(f'Corners: {dispargmax(corners)}')
    print('------------------------------------------------------------------')

def loaddata(fullpath):
    with open(fullpath,"r") as f:
        X = []
        for x in f:
            xx = x.split(',')
            X.append([float(xi) for xi in xx])
    return a(X)

if __name__ == '__main__': # Use the code on an example
    # import examples as ex
    X = a([[1.0, 9.0], # X = ex.D1
        [1.125, 8.25],
        [1.25, 7.5],
        [1.375, 6.75],
        [1.5, 6.0],
        [1.625, 5.25],
        [1.75, 4.5],
        [1.875, 3.75],
        [2.0, 3.0]])
    n=X.shape[0]
    nn=a(list(range(n)))
    print(X.shape)
    mw,p_0,t_0 = Centre(X)
    mids = (X[:,1] + X[:,0])/2

    print('  ')
    print('  ')
    vq, pq, real_var_q, corners_x_q, f_count, g_count, h_count = quadratic_algorithm(X)
    tq = Threshold(X,pq)
    print('##################################################################')
    print('##################### QUADRATIC ALGORITHM ########################')
    print('##################################################################')
    print(f'Number of function evaluations v(p): {f_count}')
    print(f'Number of gradient evaluations g(p): {g_count}')
    print(f'Number of hessian evaluations  h(p): {h_count}')
    print('------------------------------------------------------------------')
    print(f'Variance at optimum: {vq}')
    print(f'Optimum distribution: {pq}')
    print(f'Threshold at optimum: {tq}')
    print(f'real valued variance:{real_var_q}')
    print('##################################################################')
    print('##################################################################')

    COUNTER_VAR = []
    COUNTER_GRAD = []
    COUNTER_HESS = []
    print('  ')
    print('  ')
    v, p, real_var, corners_x__, h_mask, k, t, f_count, g_count, h_count = fast_algorithm(X,stop=1_000,e=1e-12)
    print('####################################################################')
    print('######################## FAST ALGORITHM  ###########################')
    print('####################################################################')
    print(f'Number of function evaluations v(p): {f_count}')
    print(f'Number of gradient evaluations g(p): {g_count}')
    print(f'Number of hessian evaluations h(p): {h_count}')
    print('------------------------------------------------------------------')
    print(f'Variance at optimum:  {v}')
    print(f'Optimum distribution: {p}')
    print(f'Threshold at optimum: {t}')
    print(f'real valued variance: {real_var}')
    print('####################################################################')
    print('####################################################################')