import pandas as pd

'''
a=[1,2,3,4]
myvar=pd.Series(a)
print(myvar)

#accessing elements in series
print(a[2])
#accessing multiple elements
print(myvar[[2,0]])

#Create a Series from a list of your favorite fruits and print the second and fourth elements.
fruit_list= ['Apple','Orange','Guava','Banana','Pineapple']
fruit_list_series = pd.Series(fruit_list)
print(f"Fruit List converted to Series: {fruit_list_series}")
print(fruit_list_series[[1,3]])

#DataFrames: 2D labeled data structure with row and columns
#similar to spreadsheet of SQL

data= {'Name':['Alice','Bob','Charlie'],
       'Age':[25,30,35],
       'City':['New York','Los Angeles','Chicago']
       }
data_frame = pd.DataFrame(data)
print(data_frame)

#Accessing columns using the key
print(data_frame[['Name','City']])

#Exercise:Create a DataFrame with information about your favorite movies
#e.g., title, year, rating) and print the "Title" and "Rating" columns.
movie_dictionary ={'Title':['Exodus','Moses','The Message','Gospel','Ten Commandments'],
                   'Year':[1980,1978,1990,1993,2001],
                   'Rating':['4 star','4 star','5 star','1 star','3 star']
                   }
data_frame_movie= pd.DataFrame(movie_dictionary)


print(data_frame_movie)
#print the title and rating
print(data_frame_movie[['Title','Rating']])

#Basic Operations:
#Selecting rows and columns: LABEL BASED:
print(data_frame_movie.iloc[0])
print(data_frame_movie['Title'])

#Exercise: Select the second row of your DataFrame using both .loc[] and .iloc[].
print(data_frame_movie.loc[2])
print(data_frame_movie.iloc[2])
print(data_frame_movie['Rating'])

#Adding new columns:
data_frame_movie['Review']=['Good','Good','Very Good','Bad','Average']


#Dropping: we can drop rows or columns using the .drop method
data_frame_dropped_columns = data_frame_movie.drop(columns=['Year'])
print(data_frame_dropped_columns)
data_frame_dropped_rows= data_frame_movie.drop(index=2)
print(data_frame_dropped_rows)

#Exercise:Drop Year and Rating and then Row 3
data_frame_movie_dropped_year_rating = data_frame_movie.drop(columns=['Year','Rating'])
print(data_frame_movie_dropped_year_rating)
data_frame_movie_dropped_row_3 = data_frame_movie.drop(index=2)

#Renaming Columns: using .rename() method
data_frame_movie_renamed= data_frame_movie.rename(columns={'Title':'Movie Name','Rating':'Stars'})
print(data_frame_movie_renamed)

data_frame_movie_renamed_year= data_frame_movie.rename(columns={'Year':'Released In'})
print(data_frame_movie_renamed_year)

#Data Input and Output
data_frame_movie.to_csv('output.csv',index=False)
data_frame_movie_csv = pd.read_csv('output.csv')
print(data_frame_movie_csv.head())

#Exploring Data
#Viewing Data: .head() and .tail()
print(data_frame_movie.head())
print(data_frame_movie.tail())

#Checking Shape and Size: .shape and .size
print(data_frame_movie.shape)
print(data_frame_movie.size)

#Getting descriptive Stats:
print(data_frame_movie.describe())

#DATA MANIPULATION AND CLEANING:
#indexing and selection:
#loc[]: Label-based indexing
#iloc[]: Position-based indexing

#LABEL BASED INDEXING: loc[]
#Selecting a single row by label
print(data_frame_movie.loc[0])
#selecting multiple rows by label
print(data_frame_movie.loc[[0,2]])
#selecting specific rows and columns
print(data_frame_movie.loc[[0,2],['Title','Year']])

#POSITION BASED INDEXING: iloc[]
print(data_frame_movie.iloc[2])
print(data_frame_movie.iloc[[1,3]])
print(data_frame_movie.iloc[[0,2],[1,2]])

#Practice Exercise: 1. use .loc[] 2. use iloc[]
print(data_frame_movie.loc[[0,1],['Year']])
print(data_frame_movie.iloc[[2,3],[0,1]])

#=====================================================
#HANDLING MISSING DATA:
#=====================================================
#types: 1. NaN - used for numerical data
# 2. None - used for non Numerical data
# 3. NaT - Not a Time: used for missing datetime values

#Identify Missing Data: before handling missing data, you need to identify
#where it exists in your dataset
#METHODS: isnull(): return true for missing values
# notnull(): returns true for non missing values
# info(): prpvides a summary of the dataset, including the count of non-null values in each column
#----------------------------------------------
handle_missing_data_dictionary= {'Name':['Alice','Bob',None,'Charlie',None,'Dom'],
                                 'Age':[20,None,None,43,None,54],
                                 'Salary':[None,5000,3000,5000,3400,None]}
#converting into Dataframes:
handle_missing_data_df= pd.DataFrame(handle_missing_data_dictionary)
print(handle_missing_data_df)

#Check for missing values:
print(handle_missing_data_df.isnull())
print(handle_missing_data_df.info())

#Remove Missing Data: dropna(): removes rows and columns for missing values
#drop Rows with missing values
#print(handle_missing_data_df.dropna())

#drop columns with any missing value
print(handle_missing_data_df.dropna(axis=1))

#Filling missing datas:
#filling missing datas with 0
filling_missing_values = handle_missing_data_df.fillna(0)
print(filling_missing_values)

#filling missing values with the mean of the column
handle_missing_data_df['Age'] = handle_missing_data_df['Age'].fillna(handle_missing_data_df['Age'].mean())
print(f"Missing data filled with mean of the desired column:\n {handle_missing_data_df}")
'''
#DATA TRANSFORMATION:
#=========================================
#Changing Data Types: using .astype()
data_transformation_dictionary = {'Name':['James','Jakob','John','Jerry','Jasmine'],
                                  'Age':[20,25,30,35,40],
                                  'Address':['India','Pakistan','Bangladesh','India','China'],
                                  'Salary':[10000,20000,25000,23000,14000]}

#converting into dataframes
data_transformation_dictionary_df = pd.DataFrame(data_transformation_dictionary)
'''
#transforming int dtype into float
data_transformation_dictionary_df['Age'] = data_transformation_dictionary_df['Age'].astype(float)
print(data_transformation_dictionary_df)
print(data_transformation_dictionary_df.dtypes)

#converting int into string
data_transformation_dictionary_df['Salary']=data_transformation_dictionary_df['Salary'].astype(str)
print(data_transformation_dictionary_df.dtypes)

#Applying Functions: .apply() , .map() , .applymap()

#Applying a function to a column
data_transformation_dictionary_df['Age in 10 years']= data_transformation_dictionary_df['Age'].apply(lambda x:x+10)
print(data_transformation_dictionary_df)

#Map values in a column:
data_transformation_dictionary_df['Address']=data_transformation_dictionary_df['Address'].map({'India':'Mexico','Bangladesh':'S.Africa'})
print(data_transformation_dictionary_df)

#Apply a function to the entire DataFrame:
data_transformation_dictionary_df = data_transformation_dictionary_df.applymap(lambda x: x.upper() if isinstance(x,str) else x)
print(data_transformation_dictionary_df)

#PRACTICE EXERCISE:
data_transformation_dictionary_df['Age_In_5_Years'] = data_transformation_dictionary_df['Age'].apply(lambda x:x+5)
print(data_transformation_dictionary_df)

data_transformation_dictionary_df['Address']=data_transformation_dictionary_df['Address'].map(lambda x: 'IND' if (x=='India') else x)
print(data_transformation_dictionary_df)

#SORTING AND RANKING:
#----------------------------------
#sorting by column values: .sort_values()

#Sort by 'Age' in ascending order
sorted_data = data_transformation_dictionary_df.sort_values(by='Age',ascending=True)
print(f"Sorted data in ascending order: \n{sorted_data}")

#Sort by 'Age' in descending order
sorted_data = data_transformation_dictionary_df.sort_values(by='Age',ascending=False)
print(f"Sorted data in ascending order: \n{sorted_data}")

#sort by multiple columns:
sorted_multi_col = data_transformation_dictionary_df.sort_values(by=['Age','Salary'],ascending=[False,True])
print(sorted_multi_col)

#Sort by index: .sort_index()
df_sorted_index = data_transformation_dictionary_df.sort_index()
print(df_sorted_index)

#Ranking Data:
#ranking the salary column
data_transformation_dictionary_df['Salary_Rank']=data_transformation_dictionary_df['Salary'].rank()
print(data_transformation_dictionary_df)

data_transformation_dictionary_df=data_transformation_dictionary_df.sort_values(by='Salary',ascending=False)
print(data_transformation_dictionary_df)

data_transformation_dictionary_df['Age Rank']=data_transformation_dictionary_df['Age'].rank()
print(data_transformation_dictionary_df)
'''
#Grouping and Aggregation:
#----------------------------------------
















