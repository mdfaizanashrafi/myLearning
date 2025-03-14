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

x_sub2=np.array([0,1,2,3])
y_sub2=np.array([3,4,6,7])

plt.subplot(2,1,1)
plt.plot(x_sub2,y_sub2)
plt.title("Sales")

x2_sub2=np.array([0,1,2,3])
y2_sub2=np.array([10,20,30,40])

plt.subplot(2,1,2)
plt.plot(x2_sub2,y2_sub2)
plt.title("Income")

plt.suptitle("Goods and Services")
plt.show()

#===========================================
#MATPLOTLIB SCATTER:
#==========================================
#Creating Scatter plots:
#-----------------------
x_scatter=np.array([1,2,3,4,5,6,7,8,9,10])
y_scatter=np.array([5,3,6,9,24,2,3,7,13,24])

plt.scatter(x_scatter,y_scatter)

#Compare plots:
#day 1
x_comp=np.array([1,2,3,4,5,6,7,8,9,10])
y_comp=np.array([5,3,6,9,24,2,3,7,13,24])

plt.scatter(x_comp,y_comp,c='hotpink')

#day 2
x_comp2=np.array([9,3,2,14,15,6,28,9,10,23])
y_comp2=np.array([5,7,4,2,6,17,4,34,3,9])

plt.scatter(x_comp2,y_comp2,color='green1')

plt.show()

#=============================================
#Matplotlib Bars:
#----------------------
x_bar= np.array(["A","B","C","D","E","F","G","H","I","J"])
y_bar=np.array([3,4,5,6,7,8,9,10,11,12])

plt.barh(x_bar,y_bar,color='green',height=0.5)
plt.show() 

#=============================================
# #Matplotlib Histogram:
# ----------------------
x_hist= np.random.normal(170,10,250)
plt.hist(x_hist)
plt.show()
'''
#Matplotlib Pie Chart:
#---------------------
y_pie=np.array([30,25,20,40])
my_labels=["A","B","C","D"]
myexoplode=[0,0.1,0,0]

plt.pie(y_pie,labels=my_labels,startangle=90,explode=myexoplode,shadow=True)
plt.legend(title="Legend Title")
plt.show()