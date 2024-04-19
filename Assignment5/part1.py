#!/usr/bin/env python3
import numpy as np
randn_generator = np.random.default_rng()
from sklearn.linear_model import LinearRegression
np.random.seed(0)


def show_correlation(x1_vec , x2_vec , x3_vec):
    coeff_12 = np.corrcoef(x1_vec , x2_vec)[0 , 1]
    coeff_23 = np.corrcoef(x2_vec , x3_vec)[0 , 1]
    coeff_31 = np.corrcoef(x3_vec , x1_vec)[0 , 1]
    print(f"correlation coefficient between x1 and x2: {coeff_12:.3f}")
    print(f"correlation coefficient between x2 and x3: {coeff_23:.3f}")
    print(f"correlation coefficient between x1 and x3: {coeff_31:.3f}" )

def show_beta(beta0 , beta1 , beta2 , beta3 , model , show_3 = False):   
    print('(data generation , regression)')
    print(f'beta0: ({beta0} , {model.intercept_:.3f})')
    print(f'beta1: ({beta1} , {model.coef_[0]:.3f})')
    print(f'beta2: ({beta2} , {model.coef_[1]:.3f})')
    if show_3:
        print(f'beta3: ({beta3} , {model.coef_[2]:.3f})')


def generate_error(size , mean = 0 , std = 1):
    return np.random.normal(loc = mean , scale = std , size = size)


def run_experiment(x1_vec , x2_vec , x3_vec):
    assert len(x3_vec.shape) == 1
    assert x1_vec.shape == x2_vec.shape == x3_vec.shape
    show_correlation(x1_vec = x1_vec , x2_vec = x2_vec, x3_vec = x3_vec)
    print('________________Generating y___________________________________________________')
    (beta0 , beta1 , beta2 , beta3) = (3 , -2 , 1 , 1.5)
   
    y_vec = beta0 + beta1 * x1_vec + beta2 * x2_vec + beta3 * x3_vec + generate_error(size = x1_vec.shape[0])
    print(f"y size: {y_vec.shape}")

    print('________________Running Regression including x3___________________________________________________')
    X = np.column_stack((x1_vec, x2_vec, x3_vec))
    model = LinearRegression()
    model.fit(X, y_vec)
    # show_beta(beta0 , beta1 , beta2 , beta3 , model , True)
    

    print('________________Running Regression excluding x3___________________________________________________')
    X = np.column_stack((x1_vec , x2_vec))
    model2 = LinearRegression()
    model2.fit(X , y_vec)
    # show_beta(beta0 , beta1 , beta2 , beta3 , model)
    print(f'x1: {beta1} , {model.coef_[0]:.3f} , {model2.coef_[0]:.3f}')
    print(f'x2: {beta2} , {model.coef_[1]:.3f} , {model2.coef_[1]:.3f}')




if __name__ == '__main__':
    data_size = 500
    x1 = np.random.uniform(low = -1 , high = 1 , size = data_size)
    x2 = np.random.uniform(low = -1 , high = 1 , size = data_size)

    # print("______________x3 positively correlated with x1____________________________________________________")

    # x3 = 0.6 + 0.95 * x1 + 0 * x2 + generate_error(size = data_size) 
    # run_experiment(x1_vec = x1 , x2_vec = x2 , x3_vec = x3)

    print("\n\n__________x3 negatively correlated with x1____________________________________________________")

    x3 = 0.6 - 0.95 * x1 + 0 * x2 + generate_error(size = data_size)
    run_experiment(x1_vec = x1 , x2_vec = x2 , x3_vec = x3)


