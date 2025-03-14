import matplotlib.pyplot as plt
import numpy as np

'''
x=np.array([80,85,89,93,96,99,104,110,120,124])
y=np.array([200,230,240,260,290,321,322,345,367,256])

font1={'family':'serif',
       'color':'blue',
       'size':20}
font2={'family':'serif',
       'color':'darkred',
       'size':15}

plt.title("Sports watch data",fontdict=font1)
plt.xlabel("Average Pulse",fontdict=font2)
plt.ylabel("Calorie Burnage",fontdict=font2)

plt.plot(x,y)
plt.grid(color='green',ls='--',linewidth=0.5)
plt.show()
'''
#subplots:
#----------------
#plot 1
x_sub=np.array([0,1,2,3])
y_sub=np.array([3,4,6,7])
plt.subplot(1,2,1)  #syntax: plt.subplot(nrows,ncols,index)
plt.plot(x_sub,y_sub)

#plot 2
x2_sub=np.array([0,1,2,3])
y2_sub=np.array([10,20,30,40])
plt.subplot(1,2,2)
plt.plot(x2_sub,y2_sub)

plt.show()
