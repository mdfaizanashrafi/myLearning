#SciPy is a scientific computation lib thta uses NumPy underneath
from matplotlib import lines
import numpy as np
#constants in scipy
from scipy import constants
from scipy.optimize import curve_fit, differential_evolution, minimize, least_squares, fsolve
from scipy.optimize import root
import matplotlib.pyplot as plt
'''
print(constants.liter)

#constant unites: a list of all units under the constants moduke can be seen using the fir()
#list all constants:
print(dir(constants))
print(constants.tera)
print(constants.ounce)
print(constants.hour)
print(constants.Julian_year)
print(constants.inch)
print(constants.mil)
print(constants.astronomical_unit)
print(constants.bar)

#scipy.optimize.minimize
result=minimize(lambda x:x**2,x0=[1.0],method='BFGS')
print("Minimum at X=",result.x)

def objective(x):
    return x**2 + 3*x +4

x0=[0.0]
result=minimize(objective,x0,method='BFGS')
print(f"Minimum value of the function: {result.fun}")  #.fun is used to get the value of the function at the optimal solution
print(f"Optimal x: {result.x}")  #.x is used to get the optimal solution

#scipy.optimize.curve_fit: used for curve fitting to find parameters that bets fit a model to data
#fit a linear model
#-----------------------------------------------------------
#generate a noisy data
x=np.linspace(0,10,100)  #Syntax: np.linspace(start,stop,num)
y=2*x+1+ np.random.normal(0,1,100) #syntax: np.random.normal(loc,scale,size)

#define the model (y=a*x+b)
def linear_model(x,a,b):
    return a*x+b

#fit the model to data
params,covariance = curve_fit(linear_model,x,y) #syntax : curve_fit(model,x,y)
#two variables because curve_fit returns two outputs, a. parameters b. covariance matrix
#params: a list/array of the optimized parameters a and b   
#covariance: a 2D array of the covariance matrix describing the uncertainty in the estimated parameters
print(f"Fitted parameters (a,b): {params}")

#scipy.optimize.least_squares: solves non-linear least squares problems.
#-----------------------------------------------------------------------
#mimimizes tge sum of squared residuals.

#define a system of equations:
def equations(vars): #fn takes a list of variables
    x,y = vars   #unpack the variables
    return [x**2 + y**2 -1, x+y-1]

#initial guesses for the variables
x0= [0.5,0.5] 
#solve using th eleast_squares
result= least_squares(equations,x0)  #minimize the sum of squared residuals returned by the equations fn.
print(f"Solution (x,y): {result.x}") #result.x means the optimal solution for the variables x and y

#Global Optimization: differential_evolution
#------------------------------------------------
def noisy_function(x):
    return x**2 + np.random.normal(0,1)

#define bounds for x
bounds=[(-5,5)]
#perform global optimization
result= differential_evolution(noisy_function,bounds)
print(f"Global Maximum x: {result.x}")
print(f"Minimum Value: {result.fun}")

#Maximizinf a Function
#-------------------------------------------------
def objective(x):
    return -((x-2)**2 +3)
#inital guess
x0=[0.0]

#perform minimiztion:
result=minimize(objective,x0,method='BFGS')
print("Maximum Value of f(x)", -result.fun)
print(f"Optimal x: {result.x}")

#============================================================
#ROOTS OF AN EQUATION:
#==============================================================
#roots or zeros of a function are the values of x where the fn=0, f(x)=0

#scipy.optimize.roots: finds the roots of a scalar or vector function
#scipy.optimize.fsolve: finds the roots of a non-linear system of equations

#single variable equation: f(x)=x^2-4=0
#---------------------------------------------
def f(x):
    return x**2 -4
x0=[1.0]
solution=root(f,x0)  #syntax: root(fn,initial guess)
print(f"Roots: {solution.x}")

#Multi-Variable System of Equations: f(x,y)=0
#--------------------------------------------------
#define ths system of equations
def equations(vars):
    x,y =vars
    eq1=x**2 + y**2 - 1  #equation for circle: x^2 + y^2 = 1
    eq2= x+y-1  #line equation: x+y=1
    return[eq1,eq2]

x0=[0.5,0.5]  #inital guess
solution=root(equations,x0) #syntax: root(fn,inital)
print(f"Roots (x,y): {solution.x}")

#Visualization: of circle and line
#-------------------------------------------
#linspace: generates linearly spaced vectors or arrays.
#circle equation: x^2 + y^2 =1
theta = np.linspace(0,2*np.pi,100)  #creates an array,len(100) of angles from 0 to 2pi(one full rotation around circle)
x_circle=np.cos(theta)  #contains 100 points on x-coordinates of the circle
y_circle=np.sin(theta)  #contains 100 pounts on y-coordinates of the circle

#line equation: x+y=1
x_line=np.linspace(-1,2,100)  #creates 100 evenly spcaed value btw -1 and 2
y_line = 1-x_line  #for each value of x, we comupte y using the line equation

#plot the circle and th eline
plt.figure(figsize=(6,6)) #creates a figure with a size of 6 inches by 6 inches
plt.plot(x_circle,y_circle,label="Circle: $x^2 + y^2 =1$",color='blue')
plt.plot(x_line,y_line,label="Line: $x+y=1$",color='orange')

#mark the intersection points
intersection_1=[1.0,0.0]
intersection_2=[0.0,1.0]
plt.scatter([intersection_1[0],intersection_2[0]],
[intersection_1[1],intersection_2[1]],
color='red',label='Intersection Points')  #syntax: plt.scatter(x,y,color=color,label=label,marker=marker)

#add labels and legends
plt.axhline(0,color='black',linewidth=0.5,linestyle='--')
plt.axvline(0,color='black',linewidth=0.5,linestyle='--')
plt.grid()
plt.legend()
plt.title("Circle and Line Intersection")
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.axis('equal')  #ensures that the asoect ratio is equal

plt.show()

#================================================
#SPARSE MATRIX: data set where most of the elements are zero.
#======================================================
#define non zero values, rows indices, col indices
data=[1,2,3,4]  #only non-zeros are stored
rows=[0,1,2,3]  #rows position of data
cols=[0,1,2,3]  #col position of data

#create sparse matrix
sparse_matrix_coo = coo_matrix((data,(rows,cols)),shape=(4,4)) #syntax: coo_matrix((data,(r,c)),shape=(m,n))
print(sparse_matrix_coo)

print(sparse_matrix_coo.toarray())  #dense representation

#Using CSR Matrix: Compressed Sparse Row:
#---------------------------------------------
#define a dense matrix
dense_matrix=[[0,0,0,0],
              [0,2,0,0],
              [0,0,3,0],
              [0,0,0,4]]

#convert to CSR format:
sparse_matrix_csr= csr_matrix(dense_matrix)
print(f"Sparse Matrix (CSR FOrmat):\n {sparse_matrix_csr}")
#access non zero elements:
print(f"Non-Zero elements:\n {sparse_matrix_csr.data}") #non-zero elemets
print(f"Col indices: \n {sparse_matrix_csr.indices}")  #indices gives col numbers with non zero value
print(f"Indptr (Row/Index Pointers): \n {sparse_matrix_csr.indptr}")

#OPERATIONS ON SPARSE MATRICES:
#------------------------------------
#matrix multiplication:
A= csr_matrix([[1,0,0],
               [0,2,0],
               [0,0,3]])
B= csr_matrix([[0,4,0],
               [5,0,0],
               [0,0,6]])

#multiply matrices
C= A.dot(B)  #multiplied matrix(C) = A.dot(B)
print(f"Result of A*B (sparese):\n {C}")
print(f"Dense Representation of Result:\n {C.toarray()}")
"""Start
  |
  v
Import csr_matrix from scipy.sparse
  |
  v
Define Sparse Matrix A:
[[1, 0, 0],
 [0, 2, 0],
 [0, 0, 3]]
  | 
  |--> Internally, A is stored in CSR format:
  |    - `data`: [1, 2, 3]
  |    - `indices`: [0, 1, 2]
  |    - `indptr`: [0, 1, 2, 3]
  |
  v
Define Sparse Matrix B:
[[0, 4, 0],
 [5, 0, 0],
 [0, 0, 6]]
  | 
  |--> Internally, B is stored in CSR format:
  |    - `data`: [4, 5, 6]
  |    - `indices`: [1, 0, 2]
  |    - `indptr`: [0, 1, 2, 3]
  |
  v
Multiply Matrices: C = A.dot(B)
  |
  |--> Internal Working of Multiplication:
  |    1. Iterate over rows of A using `indptr`.
  |    2. For each row in A, find non-zero elements.
  |    3. Multiply these with corresponding columns in B.
  |    4. Accumulate results into new sparse matrix C.
  |
  v
Print Sparse Result (C)
  |
  |--> Output:
  |    (0, 1)    4
  |    (1, 0)    10
  |    (2, 2)    18
  |
  v
Convert Sparse Result to Dense Format: C.toarray()
  |
  |--> Internal Conversion:
  |    - Fill zeros for missing indices.
  |    - Place non-zero values at their respective positions.
  |
  v
Print Dense Representation of Result
  |
  |--> Output:
  |    [[ 0  4  0]
  |     [10  0  0]
  |     [ 0  0 18]]
  |
  v
End"""

#Converting between Formats:  .tocsr()  .tocsc()
#--------------------------------------------------
#create a COO matrix:
coo_matrix_eg=coo_matrix([[1,0,0],
                       [0,2,0],
                       [0,0,3]])
#convert to CSR format and CSC format
csr_conversion = coo_matrix_eg.tocsr()
csc_conversion= coo_matrix_eg.tocsc()
print(f"COO Matrix:\n {coo_matrix_eg}")
print(f"Converted to CSR:\n {csr_conversion}")
print(f"Converted to CSC:\n {csc_conversion}")

#===================================================
#SCIPY GRAPHS:
#===================================================
#Adjecent Matrix: adjecent matrix is like a map that show how things are connected

adjecent_matrix=[[0,1,1,0], #A talks to BnC
                 [1,0,0,1], #B talks to AnD
                 [1,0,0,1], #C talks to AnD
                 [0,1,1,0]] #D talks to BnC

#convert to a sparse matrix for efficient storage
convert2sparse=csr_matrix(adjecent_matrix)
print(convert2sparse)

#connected componenets
num_componenets, labels=connected_components(convert2sparse)
print(f"Number of groups of friends: {num_componenets}")
print(f"Which group each friend belongs to: {labels}")

#Dijkstra: method to find the shortest path in a graph from one element to annother
#----------------------------------------------------------------------------------------
#Syntax: dijkstra(csr_matrix,return_predecessors,indices)
dijkstra_arr = np.array([[0,1,2],
                         [1,0,0],
                         [2,0,0]])
dijkstra_matrix= csr_matrix(dijkstra_arr)
print(dijkstra(dijkstra_matrix,return_predecessors=True,indices=0))
'''
#Floyd Warshall: used to find shortest path between all pairs of elements
#---------------------------------------------------------------------------
foyd_arr=np.array([[0,1,2],
                   [1,0,0],
                   [2,0,0]])
floyd_matrix=csr_matrix(foyd_arr)
print(floyd_warshall(floyd_matrix,return_predecessors=True))
































