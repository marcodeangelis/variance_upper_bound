import compute
from compute import (a,np)

D1  = a([[3.5,6.4], # Puffy data set
         [6.9,8.8],
         [6.1,8.4],
         [2.8,6.7],
         [3.5,9.7],
         [6.5,9.9],
         [0.15,3.8],
         [4.5,4.9],
         [7.1,7.9]])

D2 = a([[1.0, 9.0],
        [1.125, 8.25],
        [1.25, 7.5],
        [1.375, 6.75],
        [1.5, 6.0],
        [1.625, 5.25],
        [1.75, 4.5],
        [1.875, 3.75],
        [2.0, 3.0]   ])

D3 = a([[-0.75, -0.5, -0.25,  -0.05, 0., 0.05, 0.25, 0.5, 0.75, 1., 1.125, 1.25,  1.375, 1.5, 1.625, 1.75, 1.875, 2., 2.25, 2.495],
       [10.75, 10.5, 10.25, 10.05, 10., 9.95, 9.75, 9.5, 9.25, 9., 8.25, 7.5, 6.75, 6., 5.25, 4.5, 3.75, 3., 2.75, 2.505]]).T 

if __name__ == '__main__':

    print('#################### D1 ###################')
    print('############# TABLE 1 -- pg 8 #############')
    X = D1
    vx, corners_x_exact, fx_count = compute.exact_algorithm_2n(X)
    compute.COUNTER_VAR,compute.COUNTER_GRAD,compute.COUNTER_HESS = [],[],[]
    v, p, real_var, corners_x_, h_mask, k, t, vp_count, g_count, h_count = compute.fast_algorithm(X,stop=1_000,e=1e-9)
    compute.COUNTER_VAR,compute.COUNTER_GRAD,compute.COUNTER_HESS = [],[],[]
    exact = r'$f(x^*)$'+f' = {vx} <- Exact '
    print(exact)
    polyn = r'$v(p^*)$'+f' = {v} <- Sharp bound'
    print(polyn)
    print(f'K = {k}')
    hh = sum(h_mask)
    print(f'#H = {hh}')
    print(f'Number of f(x) calls: {fx_count}')
    print(f'Number of v(p) calls: {vp_count}')
    print('###################################\n')
    # #################### D1 #######################
    # ############# TABLE 1 -- pg 8 #################
    # First threshold interval contains 0 midpoint(s).
    # Exact bound was found.
    # $f(x^*)$ = 10.974444444444444 <- Exact 
    # $v(p^*)$ = 10.974444444444442 <- Sharp bound
    # K = 0
    # #H = 0
    # Number of f(x) calls: 512
    # Number of v(p) calls: 1
    # #################################################


    print('#################### D2 ###################')
    print('############# TABLE 1 -- pg 8 #############')
    X = D2
    vx, corners_x_exact, fx_count = compute.exact_algorithm_2n(X)
    compute.COUNTER_VAR,compute.COUNTER_GRAD,compute.COUNTER_HESS = [],[],[]
    v, p, real_var, corners_x_, h_mask, k, t, vp_count, g_count, h_count = compute.fast_algorithm(X,stop=1_000,e=1e-9)
    compute.COUNTER_VAR,compute.COUNTER_GRAD,compute.COUNTER_HESS = [],[],[]
    exact = r'$f(x^*)$'+f' = {vx} <- Exact'
    print(exact)
    polyn = r'$v(p^*)$'+f' = {v} <- Sharp bound'
    print(polyn)
    print(f'K = {k}')
    hh = sum(h_mask)
    print(f'#H = {hh}')
    print(f'Number of f(x) calls: {fx_count}')
    print(f'Number of v(p) calls: {vp_count}')
    print('####################################\n')
    # #################### D2 ###################
    # ############# TABLE 1 -- pg 8 #############
    # First threshold interval contains 1 midpoint(s).
    # Solution found in closed form becuase #H=1. Gradient task not needed.
    # Closed form solution = [0.68604651] for element # [3]
    # $f(x^*)$ = 9.725694444444445 <- Exact
    # $v(p^*)$ = 9.760850694444443 <- Sharp bound
    # K = 0
    # #H = 1
    # Number of f(x) calls: 512
    # Number of v(p) calls: 1
    # ####################################

    print('#################### D3 ###################')
    print('############# TABLE 1 -- pg 8 #############')
    X = D3
    vx, corners_x_exact, fx_count = compute.exact_algorithm_2n(X)
    compute.COUNTER_VAR,compute.COUNTER_GRAD,compute.COUNTER_HESS = [],[],[]
    v, p, real_var, corners_x_, h_mask, k, t, vp_count, g_count, h_count = compute.fast_algorithm(X,stop=1_000,e=1e-9)
    compute.COUNTER_VAR,compute.COUNTER_GRAD,compute.COUNTER_HESS = [],[],[]
    exact = r'$f(x^*)$'+f' = {vx} <- Exact'
    print(exact)
    polyn = r'$v(p^*)$'+f' = {v} <- Sharp bound'
    print(polyn)
    print(f'K = {k}')
    hh = sum(h_mask)
    print(f'#H = {hh}')
    print(f'Number of f(x) calls: {fx_count}')
    print(f'Number of v(p) calls: {vp_count}')
    print('####################################\n')
    # #################### D3 ###################
    # ############# TABLE 1 -- pg 8 #############
    # First threshold interval contains 10 midpoint(s).
    # There are 10 midpoints of equal value. The gradient task is needed.
    # Gradient ascent begins.. 
    # Gradient ascent stopped reaching the desired accuracy, after K=21.
    # $f(x^*)$ = 17.588838687499994 <- Exact
    # $v(p^*)$ = 17.589001250000003 <- Sharp bound
    # K = 21
    # #H = 10
    # Number of f(x) calls: 1048576
    # Number of v(p) calls: 2
    # ####################################

    times = [1,7,23,41,79,117]
    print('#################### D1 ###################')
    print('############# TABLE 2 -- pg 8 #############')
    KK = []
    for j in range(len(times)):
        X = D1
        X = np.tile(X,(times[j],1))
        compute.COUNTER_VAR,compute.COUNTER_GRAD,compute.COUNTER_HESS = [],[],[]
        v, p, real_var, corners_x_, h_mask, k, t, vp_count, g_count, h_count = compute.fast_algorithm(X,stop=1_000,e=1e-9)
        KK.append(k)
    print(f'Number of iterations K: {KK}')
    print('###################################\n')
#     Number of iterations K: [0, 0, 0, 0, 0, 0]

    print('#################### D2 ###################')
    print('############# TABLE 2 -- pg 8 #############')
    KK = []
    for j in range(len(times)):
        X = D2
        X = np.tile(X,(times[j],1))
        compute.COUNTER_VAR,compute.COUNTER_GRAD,compute.COUNTER_HESS = [],[],[]
        v, p, real_var, corners_x_, h_mask, k, t, vp_count, g_count, h_count = compute.fast_algorithm(X,stop=100_000,e=1e-9)
        KK.append(k)
    print(f'Number of iterations K: {KK}')
    print('###################################\n')
#     Number of iterations K: [0, 467, 854, 1144, 1591, 1938]

    print('#################### D3 ###################')
    print('############# TABLE 2 -- pg 8 #############')
    KK = []
    for j in range(len(times)):
        X = D3
        X = np.tile(X,(times[j],1))
        compute.COUNTER_VAR,compute.COUNTER_GRAD,compute.COUNTER_HESS = [],[],[]
        v, p, real_var, corners_x_, h_mask, k, t, vp_count, g_count, h_count = compute.fast_algorithm(X,stop=100_000,e=1e-9)
        KK.append(k)
    print(f'Number of iterations K: {KK}')
    print('###################################\n')
#     Number of iterations K: [21, 77, 149, 202, 285, 349]