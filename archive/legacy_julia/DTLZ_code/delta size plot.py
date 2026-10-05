import matplotlib.pyplot as plt
import numpy as np
from collections import Counter 
import pandas as pd

# df = pd.read_csv("I:\\AA gourp\\nonlinear algorithm\\DTLZ(2,M) total Sp 240 time test with different numbers of dimensions_1.csv") # lab windows computer results
# df = pd.read_csv("/Users/hongxuan/Library/CloudStorage/GoogleDrive-hongxuan@umich.edu/My Drive/AA gourp/nonlinear algorithm/DTLZ(2,M) total Sp 240 time test with different numbers of dimensions.csv") # lab windows computer results
plt.figure(figsize=(6,6))
x = [1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001]
y1 = [9,4,6,20,20,20,20,20,20,20]
y2 = [0,0,0,13,20,20,20,20,20,20]
y3 = [0,0,0,2,13,20,20,20,20,20]
y4 = [0,0,0,1,4,20,20,20,20,20]
y5 = [3,0,0,0,0,20,15,20,20,16]
# plt.plot(x, y)
# plt.scatter(x, y1)
# plt.scatter(x, y2)
# plt.scatter(x, y3)
plt.plot(-np.log10(x), y1)
plt.plot(-np.log10(x), y2)
plt.plot(-np.log10(x), y3)
plt.plot(-np.log10(x), y4)
plt.plot(-np.log10(x), y5)
# plt.plot(x, y2)
# plt.plot(x, y3)
# plt.plot(x, y4)
# plt.plot(x, y5)
# labels = ["DTLZ(2,3) win", "DTLZ(2,3) M1","DTLZ(5,7) dt=0.05","DTLZ(7,10)", "DTLZ(7,10) dt=0.05", "DTLZ(7,12)", "DTLZ(7,16)", "DTLZ(7,20)"]
# labels = ["DTLZ(2,3) win", "DTLZ(2,3) M1","DTLZ(5,7) dt=0.05","DTLZ(7,10)", "DTLZ(7,10) dt=0.05", "DTLZ(7,12)", "DTLZ(7,16)"]
labels = ["DTLZ(3,5)", "DTLZ(4,7)","DTLZ(4,10)","DTLZ(4,12)","DTLZ(5,16)"]
plt.legend(labels=labels, loc="lower right", frameon=False)
plt.xlabel('Delta Size (-log10)', fontsize = 12)
plt.ylabel('Accurate Partition (Max 20)', fontsize = 12)
plt.ylim(-0.5, 21)
# plt.xlim(-0.40, 1.4)
plt.show()