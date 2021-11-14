
import numpy as np
import pandas as pd

#  FUNCTION
def str_column_to_float(dataset,column):
    for row in dataset:
        row[column]=float(row[column])

def findMinMax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

def normalize_dataset(dataset):
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    minmax = findMinMax(dataset)
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) /(minmax[i][1] - minmax[i][0])
    return dataset

def sum_X_dot_W_subtract_Y(x,w,y):
    return np.dot(x,w) - y

def lossFunction(sum_X_dot_W_subtract_Y, N):
    return (0.5 * np.sum(sum_X_dot_W_subtract_Y*sum_X_dot_W_subtract_Y)) / N

def dJ_dWi(x, i, sum_X_dot_W_subtract_Y ):
    if( i == 0):
        return np.sum(sum_X_dot_W_subtract_Y)
    return np.sum(np.multiply(sum_X_dot_W_subtract_Y, x[:,i]))

# RUN FUNCTION
data = pd.read_csv('winequality-white.csv',header=None).values
# Số cột, số hàng
Columns = data.shape[0]
Rows = data.shape[1]

ColumnsForTrain = int( Columns * 80 / 100)
data = normalize_dataset(data)
x = data[:ColumnsForTrain, :11]
y = data[: ColumnsForTrain, 11]

x = np.hstack((np.ones((ColumnsForTrain, 1)), x))

w = np.random.rand(12)
w_new = [0]*12
numOfIteration = 1000
learning_rate = 0.0001
for i in range(1, numOfIteration):
    print(i)

    tem = sum_X_dot_W_subtract_Y(x,w,y)
    loss = lossFunction(tem, ColumnsForTrain)
    print(loss)

    # W new
    for i in range(1,12):
        w_new[i] = w[i] - learning_rate * dJ_dWi(x,i,tem)

    # Stop
    if( loss < 1e-3):
        print("Loss:" + str(loss))
        break

    # Continue
    w = w_new

