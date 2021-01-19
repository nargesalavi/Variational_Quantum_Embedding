import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,4,5,6])
my_xticks = ['1','2','3','4','5','6']
plt.xticks(x, my_xticks)
# for l=1,w=1,d=1
# for l=1,w=2,d=1
# for l=1,w=3,d=1
# for l=1,w=4,d=1
y = np.array([0.207044,np.nan,0.206619,np.nan,np.nan,np.nan])
plt.scatter(x, y, marker='^',color='blue')
# for l=2,w=1,d=1
# for l=2,w=2,d=1
# for l=2,w=3,d=1
y = np.array([0.376935,0.326575,0.182479,np.nan,np.nan,np.nan])
plt.scatter(x, y, marker='o',color='red')
# for l=2,w=4,d=1
y = np.array([0.400412,np.nan,np.nan,0.593843,np.nan,np.nan])
plt.scatter(x, y, marker='o',color='blue')
# for l=3
# for l=4,w=1,d=1
y = np.array([np.nan,0.312526,np.nan,np.nan,np.nan,np.nan])
plt.scatter(x, y, marker='+',color='gold')
# for l=4,w=2,d=1
y = np.array([np.nan,0.375075,0.325434,np.nan,np.nan,np.nan])
plt.scatter(x, y, marker='+',color='green')
# for l=4,w=3,d=1
# for l=4,w=4,d=1

# for l=1,w=1..3,d=2
# for l=1,w=4,d=2
y = np.array([np.nan,np.nan,np.nan,0.748411,np.nan,np.nan])
plt.scatter(x, y,marker='^',facecolors='none',edgecolors='blue')
# for l=2,w=1,d=2
# for l=2,w=2,d=2
y = np.array([np.nan,np.nan,np.nan,np.nan,0.270515,0.92881])
plt.scatter(x, y,marker='o',facecolors='none',edgecolors='green')
# for l=2,w=3,d=2
# for l=2,w=4,d=2
y = np.array([np.nan,np.nan,np.nan,0.719350,np.nan,0.568995])
plt.scatter(x, y,marker='o',facecolors='none',edgecolors='blue')
y = np.array([1,1,1,1,1,2])
plt.scatter(x, y,facecolors='none',edgecolors='r',marker='o')
plt.show()