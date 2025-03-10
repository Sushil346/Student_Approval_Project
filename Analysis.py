import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the CSV file inside the 'data' folder
path = './data/adm_data.csv'
path_test = './data/adm_test_data.csv'

if os.path.exists(path) and os.path.exists(path_test):
    print(f"CSV file Imported Successfully", '\n')
else:
    print(f"CSV file not found", '\n')

data = pd.read_csv(path, delimiter=',', on_bad_lines='skip')
data_test = pd.read_csv(path_test, delimiter=',', on_bad_lines='skip')

print('The dimensions of the dataset are ', data.shape)
print('Total number of duplicate values per row ', data.duplicated().sum())
#print('NUll values in each column', '\n', data.isnull().sum())


print(data.columns)

data['Admit_Status'] = 0  # Default all to 0 (Not Admitted)

# Set top 100 as 1 (Admitted)
data.loc[ data['Chance of Admit'].rank(method='first', ascending=False) <= 300, 'Admit_Status'] = 1

# print(data['Chance of Admit','Admit_Status'])

# Selecting 390 rows for learning and rem 10 for prediction_test
X_train = data.iloc[:, :-1]
y_train = data.iloc[:, -1]

X_predict_test = data_test.iloc[:,:]

print('The shape of X_train is: ' + str(X_train.shape))
print('The shape of y_train is: ' + str(y_train.shape))
print('We have m = %d training examples' % (len(y_train)))


######################### LOGISTIC REGRESSION ####################################################
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) ###Scaled and saved there itself

m,n = X_train.shape
def sigmoid(z):
    result = (1 / (1 + np.exp(-z)))
    return result


def compute_costf(X, y, w, b):
    m,n = X.shape
    z = np.dot(X, w) + b
    fwb = sigmoid(z)

    #####   ERROR IS Encountered due to log0 and log 1 so it is added...
    # Clip values to avoid log(0) or log(1)
    fwb = np.clip(fwb, 1e-10, 1 - 1e-10)


    cost_error = - (np.dot(y, np.log(fwb)) + np.dot((1-y), np.log(1-fwb)))
    cost_error *= (1 / m)
    return cost_error

def compute_gradient(X, y, w, b):
    m, n = X.shape
    z = np.dot(X, w) + b
    fwb = sigmoid(z)

    dj_dw = (1/m) * np.dot(X.T, (fwb - y))
    dj_db =  np.mean( fwb -y )

    return dj_dw, dj_db

def gradient_descent(X, y, wi, bi, cost_fun, gradient_fun, alpha, iterator):
    m=len(X)

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []

    for i in range(iterator):
        dj_dw, dj_db = gradient_fun(X, y, wi, bi)

        wi = wi - alpha * dj_dw
        bi = bi - alpha * dj_db

        cost = cost_fun(X, y, wi, bi)
        J_history.append(cost)

        # Print cost and w every at intervals 10 times or as many iterations if < 10
        if i % (iterator // 10) == 0 or i == iterator - 1:
            cost = cost_fun(X, y, wi, bi)
            J_history.append(cost)

            w_history.append(wi)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    return wi, bi, J_history, w_history

def model_learn():
    initial_w = 0.1 * (np.random.rand(n) )
    initial_b = -8

    # Some gradient descent settings
    iterations = 100000
    alpha = 0.0005

    w,b,j_history,w_history= gradient_descent(X_train ,y_train, initial_w, initial_b, compute_costf, compute_gradient, alpha, iterations)

    return w, b, j_history, w_history
def predict(X, w, b):
    # number of training examples
    m, n = X.shape
    p = np.zeros(m)

    z_wb = np.dot(X, w) + b

    # Calculating the prediction for this example
    f_wb = sigmoid(z_wb)

    # Applying the threshold
    p = (f_wb >= 0.5).astype(int)

    return p

w,b, j_history, w_history = model_learn()
p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))

np.random.seed(1)
tmp_X = X_predict_test
tmp_X = scaler.transform(tmp_X)

tmp_p = predict(tmp_X, w, b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')


# Cost vs Iterations
plt.plot(j_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Iterations')


plt.tight_layout()
plt.show()
