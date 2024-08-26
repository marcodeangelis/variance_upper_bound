`compute.py` computes a sharp (best-possible) upper bound on the population (or sample) variance statistic. 

This code works independently of interval arithmetic libraries. 

Only Numpy is needed for this code to work. Scipy is needed for the quadratic algorithm only, i.e. to enable comparisons.

The data needed to compute the population variance must have shape (n,2), where 2 is the imprecision dimension.

Dependencies

* Numpy
* Scipy (optional, for quadratic algorithm) 
* Matplotlib Pyplot (optional, for plotting)

To use this code just run [`compute.py`](compute.py) with the interval data of choice. Note the data must have shape `(n,2)`. 

# Cite 
```
de Angelis, M. (2024). Sharp Polynomial Upper Bound on the Variance. In Combining, Modelling and Analyzing Imprecision, Randomness and Dependence. Springer Nature. 
```

## Reproducibility 
`variance.py` can be used to reproduce the results of the above paper. Just run [`smps24.py`](smps24.py) or see [smps24.md](smps24.md).
