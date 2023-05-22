import numpy as np
from sklearn.datasets import make_classification
samples = 500 # 50000 - long enough
clasData = make_classification(n_features = 5, n_samples = samples, random_state = 4)
x=clasData[0]
y=clasData[1]

learning_rate = 0.05
n_iter = 600
r = np.array([1/samples for _ in range(samples)]) # one dimentional weights are as follows
w = np.array([1/x.shape[1] for _ in range (x.shape[1])])  #an array of weights for each x variables

def x_n (r, x_t, x_t_1, y, z, beta, alpha):
    # r, x, y, z - vectors
    # alpha, beta - scalars
    return r*y + beta*(x_t-x_t_1) - alpha*z

def y_n (x_t, x_t_1, beta):
    return x_t - beta*(x_t-x_t_1)

def s_n (s_t, r): # s_i = const 
    return r*s_t

def z_n(r, z_t, grad_i, grad_i_1, s_i, s_i_1):
    return r*z_t + grad_i/s_i + grad_i_1/s_i_1

sigmoid = lambda yhat: 1/(1+np.exp(-yhat))

def f(x_t, w=w, c=x, b=y):
    s = 0
    for i in range(len(x_t)):
        s += np.log(1+np.exp((-c[i].T*x_t[i])*b[i]))
    return s + w*(x_t*x_t)

def grad(x_t, w=w, c=x, b=y):
    s = 0
    for i in range(len(x_t)):
        s += -c[i].T*b[i]*np.exp((-c[i].T*x_t)*b[i])/(1+np.exp((-c[i].T*x_t)*b[i]))
    return s + w*(x_t)


iter = 100

x_t, x_t_1, y_t, s_t, z_t, grad_t, grad_t_1 = x, x, x, 1, f(x), grad(x), grad(x)
print(f"shape(r*y)={(r*y).shape}, shape(beta*(x_t-x_t_1)) = {(0.5*(x_t-x_t_1)).shape}")
for i in range(iter):
    x_t_1 = x_t
    x_t = x_n(r, x_t, x_t_1, y_t, z_t, beta=0.5, alpha=0.01)
    y_t = y_n(x_t,x_t_1, beta=0.5)
    s_t = s_t
    grad_t_1 = grad_t
    grad_t = grad(x_t)
    z_t = z_n(r,z_t,grad_t, grad_t_1,s_t)

print(f'x = {x}\ny = {y}\nx_t = {x_t}\ny_t = {y_t}\nf_0 = {f(x)}\nf_n = {f(x_t)}\n')