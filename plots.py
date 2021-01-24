import matplotlib.pyplot as plt
#import matplotlib.axes as axes
import numpy as np

#axes.Axis.set_axisbelow(True)
x = np.array([1,2,3,4,5,6,7])
my_xticks = ['1','2','3','4','5','6','7']
plt.xticks(x, my_xticks)
# for L=1,w=1,d=1
# for L=1,w=2,d=1
# for L=1,w=3,d=1
# for L=1,w=4,d=1
y = np.array([0.207044,np.nan,np.nan,0.206619,np.nan,np.nan,np.nan])
plt.scatter(x, y, marker='^',color='blue',label='L=1,w=4,d=1')
# for l=2,w=1,d=1
# for l=2,w=2,d=1
# for l=2,w=3,d=1
y = np.array([0.376935,np.nan,0.326575,0.182479,np.nan,np.nan,np.nan])
plt.scatter(x, y, marker='o',color='red',label='L=2,w=3,d=1')
# for l=2,w=4,d=1
y = np.array([0.400412,np.nan,np.nan,np.nan,0.593843,np.nan,np.nan])
plt.scatter(x, y, marker='o',color='blue',label='L=2,w=4,d=1')
# for l=3
# for l=4,w=1,d=1
y = np.array([0.116092,0.103657,0.312526,np.nan,np.nan,np.nan,np.nan])
plt.scatter(x, y, marker='s',color='purple',label='L=4,w=1,d=1')
# for l=4,w=2,d=1
y = np.array([np.nan,np.nan,0.375075,0.325434,np.nan,0.398591,0.660803])
plt.scatter(x, y, marker='s',color='green',label='L=4,w=2,d=1')
# for l=4,w=3,d=1
# for l=4,w=4,d=1

# for l=1,w=1..3,d=2
# for l=1,w=4,d=2
y = np.array([np.nan,np.nan,np.nan,np.nan,0.748411,np.nan,np.nan])
plt.scatter(x, y,marker='^',facecolors='none',edgecolors='blue',label='L=1,w=4,d=2')
# for l=2,w=1,d=2
# for l=2,w=2,d=2
y = np.array([np.nan,np.nan,np.nan,np.nan,np.nan,0.270515,0.92881])
plt.scatter(x, y,marker='o',facecolors='none',edgecolors='green',label='L=2,w=2,d=2')
# for l=2,w=3,d=2
# for l=2,w=4,d=2
y = np.array([np.nan,np.nan,np.nan,np.nan,0.719350,np.nan,0.568995])
plt.scatter(x, y,marker='o',facecolors='none',edgecolors='blue',label='L=2,w=4,d=2')
# for l=3
# for l=4,w=2,d=2
y = np.array([0.482175,np.nan,np.nan,np.nan,np.nan,0.469099,0.398838])
plt.scatter(x, y,marker='s',facecolors='none',edgecolors='green',label='L=4,w=2,d=2')

plt.grid(b=True, which='both', color='#666666', linestyle='--')

plt.legend(bbox_to_anchor=(1.001, 1), loc='upper left')
plt.xlabel("Circuit ID", fontsize=13)
plt.ylabel("Cost Function After 300 Steps", fontsize=13)
plt.show()