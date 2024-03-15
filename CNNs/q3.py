import numpy as np
from scipy import signal

X = np.array([[1,0,-2,3,4,1],
              [2,9,5,6,0,-1],
              [0,-3,1,3,4,4],
              [6,5,2,0,6,8],
              [-5,4,-3,1,3,-2],
              [4,1,2,8,9,7]])

kernel = np.array([[-1,-1,-1],
                   [-1,8,-1],
                   [-1,-1,-1]])

print(signal.convolve2d(X, kernel, mode='valid'))