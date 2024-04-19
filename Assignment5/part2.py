#!/usr/bin/env python3
import numpy as np
randn_generator = np.random.default_rng()
from sklearn.linear_model import LinearRegression
np.random.seed(0)


def show_correlation(x1_vec , x2_vec , name_vec , name):
    coeff_12 = np.corrcoef(x1_vec , x2_vec)[0 , 1]
    coeff_23 = np.corrcoef(x2_vec , name_vec)[0 , 1]
    coeff_31 = np.corrcoef(x1_vec , name_vec)[0 , 1]
    print(f"correlation coefficient between x1 and x2: {coeff_12:.3f}")
    print(f"correlation coefficient between x2 and {name}: {coeff_23:.3f}")
    print(f"correlation coefficient between x1 and {name}: {coeff_31:.3f}" )
    print()




def show_beta(beta0 , beta1 , beta2 , beta3 , model , show_3 = False):   
    print('(data generation , regression)')
    print(f'beta0: ({beta0} , {model.intercept_:.3f})')
    print(f'beta1: ({beta1} , {model.coef_[0]:.3f})')
    print(f'beta2: ({beta2} , {model.coef_[1]:.3f})')
    if show_3:
        print(f'beta3: ({beta3} , {model.coef_[2]:.3f})')


def generate_error(size , mean = 0 , std = 1):
    return np.random.normal(loc = mean , scale = std , size = size)


def run_experiment(x1_vec , x2_vec , x3_vec , z3_vec):
    assert len(x3_vec.shape) == 1
    assert x1_vec.shape == x2_vec.shape == x3_vec.shape == z3_vec.shape
    show_correlation(x1_vec = x1_vec , x2_vec = x2_vec, name_vec = x3_vec , name = 'x3')
    show_correlation(x1_vec = x1_vec , x2_vec = x2_vec, name_vec = z3_vec , name = 'z3')

    print('________________Generating y___________________________________________________')
    (beta0 , beta1 , beta2 , beta3) = (3 , 2 , 1 , 15)
   
    y_vec = beta0 + beta1 * x1_vec + beta2 * x2_vec + beta3 * x3_vec + generate_error(size = x1_vec.shape[0])
    print(f"y size: {y_vec.shape}")


    print('________________Running Regression excluding x3 and z3___________________________________________________')
    X = np.column_stack((x1_vec , x2_vec))
    model = LinearRegression()
    model.fit(X , y_vec)
    show_beta(beta0 , beta1 , beta2 , beta3 , model)

    print('________________Running Regression using z3__________________________________________________')
    X = np.column_stack((x1_vec, x2_vec, z3_vec))
    model = LinearRegression()
    model.fit(X, y_vec)
    show_beta(beta0 , beta1 , beta2 , beta3 , model , True)
    
    print('________________Running Regression using x3__________________________________________________')
    X = np.column_stack((x1_vec, x2_vec, x3_vec))
    model = LinearRegression()
    model.fit(X, y_vec)
    show_beta(beta0 , beta1 , beta2 , beta3 , model , True)



if __name__ == '__main__':
    data_size = 50000
    print("______________using Proxy Variable______________________")
    z3 = np.random.uniform(low = -1 , high = 1 , size = data_size)
    x3 = 1.4 + 0.95 * z3 + generate_error(size = data_size)
    x1 = 1.1 + 0.4 * x3  + generate_error(size = data_size)
    x2 = 1.5 + 0.6 * x3  + generate_error(size = data_size)
    print(f'correlation coefficient between x3 and z3: {np.corrcoef(x3 , z3)[0 , 1]:.3f}')
    run_experiment(x1_vec =  x1 , x2_vec = x2 , x3_vec = x3 ,z3_vec = z3)
    
    print("______________Proxy Variable condition not satisfied!____")
    x1 = np.random.uniform(low = -1 , high = 1 , size = data_size)
    x2 = np.random.uniform(low = -1 , high = 1 , size = data_size)
    z3 = np.random.uniform(low = -1 , high = 1 , size = data_size)
    x3 = 1.4 + 0.95 * z3 + 0.5 * x1 + 0.7 * x2 + generate_error(size = data_size)
    print(f'correlation coefficient between x3 and z3: {np.corrcoef(x3 , z3)[0 , 1]:.3f}')
    run_experiment(x1_vec =  x1 , x2_vec = x2 , x3_vec = x3 ,z3_vec = z3)
