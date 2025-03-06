from itertools import count

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import pivot_table

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

#DATA TRANSFORMATION:
#=========================================
#Changing Data Types: using .astype()
data_transformation_dictionary = {'Name':['James','Jakob','John','James','Jakob'],
                                  'Age':[20,40,30,20,40],
                                  'Address':['India','Pakistan','Bangladesh','India','Pakistan'],
                                  'Salary':[10000,20000,25000,10000,20000]}

#converting into dataframes
data_transformation_dictionary_df = pd.DataFrame(data_transformation_dictionary)
data_transformation_dictionary_df['Emails']=['james43@g.co','jakob67@f.co','john34@k.co','james86@g.co','jakob2001@f.co']
print(data_transformation_dictionary_df)

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

#Grouping and Aggregation: summarize and analyze data based on certain categories
#---------------------------------------------------------------------------------

#Group by 'address' and calculate the mean age
#Syntax: df.groupby(parenthesis_for_method_call/args)[square_bracets_to_specify_column_to_workwith)
#['Age'] means take the age column from each group and
#() for calling methods like groupby and mean
#[] for indexing i.e selecting rows and columns
#groupby(): creates a group based on unique value in args column
grouped_data = data_transformation_dictionary_df.groupby('Address')['Age'].mean()
print(grouped_data)

#AGGREGATING DATA: after grouping, perform operations like sum,mean,count,etc.
#Groupby 'City' and calculate multiple stats

grouped_agg = data_transformation_dictionary_df.groupby('Address').agg({
    'Age':['mean','min','max'],
    'Salary':['sum','count']
})
print(grouped_agg)

#PIVOT TABLES: SUPER SUMMARIZER: it allows to reshape and summarize data:
#-------------------------------------------------------------

#create a pivot table:
pivot_table = pd.pivot_table(data_transformation_dictionary_df,values='Salary',index='Address',columns='Name',aggfunc='mean')
print(pivot_table)

# Plot heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5, cbar=False)
plt.title("Pivot Table: Mean Salary by Address and Name")
plt.xlabel("Name")
plt.ylabel("Address")
plt.show()

pivot_table= pd.pivot_table(data_transformation_dictionary_df,values='Salary',index='Address',columns='Name')
print(pivot_table)

#MELTING THE  PIVOT TABLE: melt(): used to reshape data from wide format to long format
#reshape the pivot table into long format

melted_df = pd.melt(pivot_table.reset_index(), #reset index to make "Adress" a column
                    id_vars='Address',  #keep "address" as an identifier
                    var_name='Name',  #name for the new "variable" column
                    value_name='Salary')  #name for the new "value" column

print(melted_df)
"""
Wide Format (Before Melting)
┌──────────────┬─────────┬─────────┬─────────┬─────────┬──────────┐
│ Address      │ James   │ Jakob   │ John    │ Jerry   │ Jasmine  │
├──────────────┼─────────┼─────────┼─────────┼─────────┼──────────┤
│ Bangladesh   │ NaN     │ NaN     │ 25000.0 │ NaN     │ NaN      │
│ China        │ NaN     │ NaN     │ NaN     │ NaN     │ 14000.0  │
│ India        │ 10000.0 │ NaN     │ NaN     │ 23000.0 │ NaN      │
│ Pakistan     │ NaN     │ 20000.0 │ NaN     │ NaN     │ NaN      │
└──────────────┴─────────┴─────────┴─────────┴─────────┴──────────┘

             Apply pd.melt(id_vars='Address', var_name='Name', value_name='Salary')
             ┌─────────────────────────────────────────────────────────────────────┐
             ▼                                                                       ▼
Long Format (After Melting)
┌──────────────┬──────────┬─────────┐
│ Address      │ Name     │ Salary  │
├──────────────┼──────────┼─────────┤
│ Bangladesh   │ James    │ NaN     │
│ Bangladesh   │ Jakob    │ NaN     │
│ Bangladesh   │ John     │ 25000.0 │
│ Bangladesh   │ Jerry    │ NaN     │
│ Bangladesh   │ Jasmine  │ NaN     │
│ China        │ James    │ NaN     │
│ China        │ Jakob    │ NaN     │
│ China        │ John     │ NaN     │
│ China        │ Jerry    │ NaN     │
│ China        │ Jasmine  │ 14000.0 │
│ India        │ James    │ 10000.0 │
│ India        │ Jakob    │ NaN     │
│ India        │ John     │ NaN     │
│ India        │ Jerry    │ 23000.0 │
│ India        │ Jasmine  │ NaN     │
│ Pakistan     │ James    │ NaN     │
│ Pakistan     │ Jakob    │ 20000.0 │
│ Pakistan     │ John     │ NaN     │
│ Pakistan     │ Jerry    │ NaN     │
│ Pakistan     │ Jasmine  │ NaN     │
└──────────────┴──────────┴─────────┘
"""
#PRACTICE EXERCISE:
#Group your DataFrame by "City" and calculate the average salary for each city.
mean_of_grouped = data_transformation_dictionary_df.groupby('Address')['Salary'].mean()
print(mean_of_grouped)

#Rank the "Age" column and add it as a new column called "Age_Rank".

data_transformation_dictionary_df['Age_Rank']=data_transformation_dictionary_df['Age'].rank()
print(data_transformation_dictionary_df)

#==============================================================
#ADVANCED LEVEL: Advanced Data Analysis and Optimization
#==============================================================

#TIME SERIES ANALYSIS:
#-------------------------

#Working with Date and Time:

#add date column:
data_transformation_dictionary_df['Date']= pd.to_datetime(['2024-01-01','2024-02-01','2024-03-01','2024-04-01','2024-05-01'])

#set date as index
data_transformation_dictionary_df.set_index('Date',inplace=True)

#resample data from monthly to quarterly frequency
quarterly_data=data_transformation_dictionary_df['Salary'].resample('QE').sum()  #QuarterlyEnd
print(quarterly_data)

#-----------------------------------------------------
#Rolling Windows:
#calculate moving avg or other stats over a sliding window

#calculate a 7-day rolling mean
rolling_mean = data_transformation_dictionary_df['Salary'].rolling(window=2).mean()
print(rolling_mean)

"""Date
2024-01-01       NaN  # Not enough data for the first row
2024-02-01    15000.0  # (10000 + 20000) / 2 = 15000
2024-03-01    22500.0  # (20000 + 25000) / 2 = 22500
2024-04-01    24000.0  # (25000 + 23000) / 2 = 24000
2024-05-01    18500.0  # (23000 + 14000) / 2 = 18500
Name: Salary, dtype: float64"""

#===========================================
#MultiIndex (Hierarchical Indexing):
#=====================================
#Creating a multiindex DataFrame
#-----------------------------------
#Creating a MultiIndex DataFrame based on Address and Name
#Allows us to create multiple levels of indexing in a DF. This is useful when we want to
#organize data hierarchically, such as grouping row by two or more categories

arrays = [
    data_transformation_dictionary_df['Address'],  #first level: Address
    data_transformation_dictionary_df['Name']      #second level: Name
]
index=pd.MultiIndex.from_arrays(arrays,names=('Address','Name')) #combines arrays into tuples
df_multi = pd.DataFrame({'Salary':data_transformation_dictionary_df['Salary'].values},
                        index=index)
#print(df_multi)

#Selecting Data from a MultiIndex:
#--------------------------------------
#select rows where 'Address' is India
print((df_multi.loc['India']))

#Stacking and Unstacking
#--------------------------------
#Stack the DataFrame to convert columns into rows

#Stack the DataFrame: process to convert columns into rows, it
#"compresses" the data by moving column labels into the index, making it more compact
stacked = df_multi.stack()
print(stacked)

#Unstacking: opposite of stacking, rows into columns, "expands" by moving index
#levels into columns, making it wider

unstacked = stacked.unstack()
print(unstacked)

#RESHAPING DATA:
#=======================
#Pivoting Data: converts data from long format to wide format.
#organizzes rows and columns based on categories, makes data easier to read

#create a pivot table
pivot_table=pd.pivot_table(data_transformation_dictionary_df,values='Salary',index='Address',aggfunc='sum')
print(pivot_table)

#melting DataFrame: converts long format into wide format
#syntax: pd.melt(DF,id_vars,value_vars,var_name,value_name)
#id_vars: identifiers,value_vars:columns to metl,var_name:new var name,value_name:name for new value column
melted=pd.melt(data_transformation_dictionary_df,id_vars=['Name'],value_vars=['Age','Salary'],var_name='Attribute',value_name='Value')
print(melted)

#PERFORMANCE OPTIMIZATION:
#===================================
#Memory Optimization:
#--------------------------
#check memory before optimization
print(data_transformation_dictionary_df.memory_usage(deep=True))

data_transformation_dictionary_df['Address']=data_transformation_dictionary_df['Address'].astype('category')
print(data_transformation_dictionary_df.memory_usage())

#Chunking Large Files:
#-------------------------
#simulate chunking by splitting the DF into smaller chunks
chunk_size=2  #no. of rows per chunk
#syntax: [df[start row:end row] for i in range(start,end,step)]
chunks= [data_transformation_dictionary_df[i:i +chunk_size].copy() for i in range(0,len(data_transformation_dictionary_df),chunk_size)]

#process each chunk:
for idx, chunk in enumerate(chunks):
    print(f"\n Processing Chunk {idx+1}:")
    print(chunk)

    #Example transformation: convert "Address" column to categorical
    chunk['Address']=chunk['Address'].astype('category')

    #perform some operation in chunk eg: avg salary
    avg_salary = chunk['Salary'].mean()
    print(f"Average Salary in Chunk {idx+1}:{avg_salary}")

#=========================================================
#ADVANCED DATA CLEANING:
#=========================================================
#Handling Duplicate Data: usedd for data cleaning and preprocessing. Helps reduce
#unnecessarily increse memory usage
#----------------------------------------------------------
#identifying duplicates
print(data_transformation_dictionary_df.duplicated()) #returns boolean

#removing duplicate data
print(data_transformation_dictionary_df.drop_duplicates())

#Handling duplicates based on specific columns
print(data_transformation_dictionary_df.duplicated(subset=['Name']))

#remove duplicates based on column
print(data_transformation_dictionary_df.drop_duplicates(subset=['Name']))

#keep unique rows, removing all duplicates
df_unique = data_transformation_dictionary_df.drop_duplicates(keep=False) #remove all rows that have duplicate, keepingn none
print(df_unique)

#counting duplicates
count_duplicates = data_transformation_dictionary_df.duplicated().sum()
print(count_duplicates)

#marking duplicates without removing them
data_transformation_dictionary_df['IsDuplicate']=data_transformation_dictionary_df.duplicated()
print(data_transformation_dictionary_df)

#STRING MANIPLATION:cleaning,transforming and extracting info from string data
#---------------------------------------------------------------------------
#accessing string method: .str accessor to apply string methods to column, vectorized

data_transformation_dictionary_df['Upper_Case_Name']=data_transformation_dictionary_df['Name'].str.upper()
print(data_transformation_dictionary_df)

#common string manipulation tasks:
#splitting strings:  .str.split(divider,expand=True)
data_transformation_dictionary_df[['Username','Domain']]= data_transformation_dictionary_df['Emails'].str.split('@',expand=True)
print(data_transformation_dictionary_df)

#replacing substrings: .str.replace()
data_transformation_dictionary_df['Domain']=data_transformation_dictionary_df['Domain'].str.replace('.co','.net',regex=False) #False inorder to treat the pattern as literal string
#regex=False: replaces exact match of '.co' with '.net'
print(data_transformation_dictionary_df)

#Extracting Patterns:  .str.extract()
# r'(\d+)': find and capture all sequences of digits in the string, \d for single digit, + for multiple digit. r: raw string, treat \ as literal string rather than escape char
data_transformation_dictionary_df['Numeric_Part']=data_transformation_dictionary_df['Emails'].str.extract(r'(\d+)',expand=False)
print(data_transformation_dictionary_df)

#Stripping Whitespace: .str.strip(), .str.lstrip(), .str.rstrip()
#removing unwanted spaces tabs,newlines from begining end our both side of a string
#spaces, tabs(\t), newlines(\n), carriage returns(\r)

# .strip()
learn_stripping="   Hey Faizan, what are you doing?!  "
cleaned_text = learn_stripping.strip() #removes whitespace from both ends of a string
print(cleaned_text)                    #doesnot affect whitespace in the middle of the string

# .lstrip(): removes white space from the left side of a string(begining)
cleaned_text_begining = learn_stripping.lstrip()
print(cleaned_text_begining)

# .rstrip(): removes white space from the right side (end) of the string
cleaned_text_end = learn_stripping.rstrip()
print(cleaned_text_end)

# Checking for Substrings:  .str.contains()  returns: boolean
data_transformation_dictionary_df['Is_G']=data_transformation_dictionary_df['Emails'].str.contains('g.co',case=False)
print(data_transformation_dictionary_df)

#COMBINING STRING COLUMNS:  .str.cat()
#combine multiple string columns into one using .str.cat()
data_transformation_dictionary_df['Name_Salary']=data_transformation_dictionary_df['Name'].str.cat(data_transformation_dictionary_df['Address'],sep=' ')
print(data_transformation_dictionary_df)
'''
#====================================================================================================
#CUSTOM DATA CLEANING PIPELINES: a pipeline that ensures that all cleaning steps are applied consistently and can be resused for new datasets
#
#STEPS TO BUILD A CUSTOM DATA CLEANING PIPELINE:
#==================================================================================================
#STEP 1: Define the raw data
data_frame_pipeline_dictionary = {
    'Name':['James','John','Jakob','Jeremiah','Johannan'],
    'Age':[25,28,None,22,26],
    'Address':['India','Pakistan','Bangladesh',None,'Pakistan'],
    'Salary':[10000,None,20000,25000,15000],
    'Email':['james@g.co',None,'jakob@co','jeremiah@co','johannan@co']
}
data_frame_pipeline_dictionary_df = pd.DataFrame(data_frame_pipeline_dictionary)
print(data_frame_pipeline_dictionary_df)

#define cleaning functions
#a) remove duplicates:
def remove_duplicates(df):
    return df.drop_duplicates()

#b) Handle missing values
def handle_missing_values(df):
    df = df.copy()
    #fill missinng ages with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    #fill missing emails with default value
    df['Email'] = df['Email'].fillna('unknown@exm.co')
    #fill missing salary with median salary
    df['Salary'] = df['Salary'].fillna(df['Salary'].median())
    #fill address with default
    df['Address'] = df['Address'].fillna('Default')
    return df

#c) Standardize the formats
def standardize_formats(df):
    df = df.copy()
    df['Name'] = df['Name'].str.title()
    df['Address'] = df['Address'].str.upper()
    df[['Username','Domain']] = df['Email'].str.split('@',expand=True)
    return df

#d) Filter rows:
def filter_rows(df):
    return df[df['Salary']>10000]

#CREATING THE PIPELINE:
def data_cleaning_pipeline(df):
    df = df.copy()
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = standardize_formats(df)
    df = filter_rows(df)
    return df

#apply the pipeline to the DataFrame
cleaned_data = data_cleaning_pipeline(data_frame_pipeline_dictionary_df)
print('\n Cleaned Data:')
print(cleaned_data)




















