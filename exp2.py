import numpy as np
from sklearn.datasets import make_classification
samples = 500 # 50000 - long enough
clasData = make_classification(n_features = 5, n_samples = samples,random_state = 4)
x=clasData[0]
y=clasData[1]

learning_rate = 0.05
n_iter = 600
b = 0
w = np.zeros(x.shape[1])  #an array of weights for each x variables

predict = lambda x, w, b: np.matmul(w,x.T) + b
sigmoid = lambda yhat: 1/(1+np.exp(-yhat))
loss = lambda y, sigmoid: -(y*np.log(sigmoid)+(1-y)*np.log(1-sigmoid)).mean()
dldw = lambda x, y, sigmoid: (np.reshape(sigmoid-y,(samples,1))*x).mean(axis = 0)
dldb = lambda y, sig: (sig-y).mean(axis = 0)
update = lambda a, g, lr: a-(g*lr)

for i in range(n_iter):
    yhat = predict(x,w,b)
    sig = sigmoid(yhat)
    grad_w = dldw(x,y,sig)
    grad_b = dldb(y,sig)
    w = update(w,grad_w,learning_rate)
    b = update(b,grad_b,learning_rate)
    
losses = []
for i in range(n_iter):
    yhat = predict(x,w,b)
    sig = sigmoid(yhat)
    logloss = loss(y,sig)
    losses.append(logloss)
    grad_w = dldw(x,y,sig)
    grad_b = dldb(y,sig)
    w = update(w,grad_w,learning_rate)
    b = update(b,grad_b,learning_rate)

import matplotlib.pyplot as plt
plt.plot(losses)


from sklearn.metrics import classification_report
yhat = predict(x,w,b)
sigy = sigmoid(yhat)
ypred = sigy >= 0.5
print(classification_report(y,ypred))

plt.show()