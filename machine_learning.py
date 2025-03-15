import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
'''
#Machine Learning: is making the computer learn from studing data and statistics.
#ML is a step into the direction of AI, wheere it analyses data and learns to predict outcomes.

#Data-Typers: 
#-------------------------
#Numerical datatypes: Continuous and Discrete
#Categorical datatypes: Nominal and Ordinal
#Text Data
#TimeSeries Data
#Image and Video Data
#Boolean Data
#Mixed datatypes

#ML: Mean Median and Mode:
#-------------------------
#Mean: average of all values
#Median: middle value
#Mode: most frequently occurring value

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
median = np.median(speed)
mean=np.mean(speed)
mode=sp.stats.mode(speed)

print(f"Mean:{mean}\n Median: {median} \n Mode: {mode}")'

#MACHINE LEARNING: STANDARD DEVIATION:
#---------------------------------------------
#it means how spread out the data is, low standard deviation meas that more data points are close to the mean
#high standard deviation means more data points are far from the mean

speed_low = [99,86,87,88,111,86,103,87,94,78,77,85,86]
low_std=np.std(speed_low)
print(low_std) #outpit: 9.25

speed_high=[32,54,999,754,44,367,65,765,343,567,567,345,456]
high_stf=np.std(speed_high)
print(high_stf)  #output: 300.74

#variance: it indicates how spread out the data is, low variance means more data points are close to the mean
#high variance means more data points are far from the mean
#If you take sqrt of variance, you get standard deviation.
#-----------------------------------------------------------------------------------

low_var= np.var(speed_low)
high_var=np.var(speed_high)
print(f"Low variance: {low_var}\n High variance: {high_var}")

#Machine Learning Percentiles:
#------------------------------------
#Percentiles are used in stats to give a quick summary of a dataset.

ages=[5,32,54,75,3,4,76,35,87,96,45,35,77,46,86,54,78,34]
ages.sort()
print(ages)
print(np.percentile(ages,70))  #syntax: np.percentile(array,percentile)

#Machine Learning: Data Distribution:
#---------------------------------------
ranndom=np.random.normal(5.0,1.0,1000000)
plt.hist(ranndom,1000) #syntax: plt.hist(array,bins)
plt.show()

#machine Learning: Linear Regression:
#---------------------------------------
#Linear Regression uses the relationship between two variables to predict the value of the third variable.

x_linreg=[5,7,8,7,54,17,2,9,76,11,25,9,6]
y_linreg=[99,86,70,98,11,54,103,87,90,78,37,85,86]

#sp.stats.linregress(x,y) used for linear regression
slope,intercept,r,p,std_err=sp.stats.linregress(x_linreg,y_linreg) #y=mx+c; m=slope,b=intercept
#r=correlation coeefficeint -1<=r<=1
#p=p-value, to test null hypothesis
#std_err=standard error, standard deviation of the error

#define prediction function
def myfunc(x_linreg):
    return slope*x_linreg+intercept #y=mx+c

#generate predicated values
my_model=list(map(myfunc,x_linreg)) #syntax: map(function,iterable)

plt.scatter(x_linreg,y_linreg)
plt.plot(x_linreg,my_model)
plt.show()
'''
#Machine Learning: Polynomial Regression:
#---------------------------------------
#Polynomial Regression is a form of linear regression where the relationship between the independent variable and the dependent variable is not linear.

x_polyreg=[5,7,8,7,54,17,2,9,76,11,25,9,6]
y_polyreg=[99,86,70,98,11,54,103,87,90,78,37,85,86]

mymodel_poly= np.poly1d(np.polyfit(x_polyreg,y_polyreg,3))
myline_poly=np.linspace(1,80,100)

plt.scatter(x_polyreg,y_polyreg)
plt.plot(myline_poly,mymodel_poly(myline_poly))
plt.show()






