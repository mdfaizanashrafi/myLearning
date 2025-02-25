

characters = {"Goku":5000,"Naruto" :2000,"Luffy":1500} #Dictionary for anime char power level

print(characters["Goku"]) #Output 5000 #Print
characters["Faizan"]=9000 #adding in dictionary
characters["Luffy"]=1200  #updating in dictionary
del characters["Naruto"]  #deleting an entry
characters.keys()  #returns keys
characters.values()  #returns value
characters.items()  #returns items
print(characters.keys())
print(characters.values())
print(characters.items())\

#exercise 1 from deepseek

grades = {
    "Alice":95,
    "Bob":88,
    "Charlie":90
}
print(grades["Charlie"])
grades["Bob"]=90  #modified
print(grades.items())
print(grades.keys())
print(grades.values())

#exercise 2 from deepseek

inventory={
    "swords":1,
    "potions":3,
    "gold":50
}
print(inventory)
inventory["Arrows"]=10  #adding a key-value pair
print(inventory)
del inventory["gold"]  #removing a key-value pair
print(inventory)
print("Swords" in inventory) #checks if something exist in dictionary
#only checks keys not values using 'in' function

#alternative way to check using .get() function
# .get() Checks for the key and returns its value (or None if missing).
sword_count = inventory.get("swords")
if sword_count is None:
    print("sword doesnt exist")
else:
    print("Sword exist")

# Exercise 1: Merge Two Dictionaries
# Create two dictionaries:
# dict1 = {"a": 10, "b": 20}
# dict2 = {"b": 30, "c": 40}
#
# Merge them into a new dictionary merged_dict where:
#
# If a key exists in both, sum their values.
#
# Otherwise, keep the key-value pair.
dict1 = {"a":10, "b":20}
dict2 = {"b":30, "c":40}

merged_dict = dict1.copy()  #copying in order to avoid modifying the original dict1
for key, value in dict2.items(): #going thru dict2 one by one
    if key in merged_dict:  #if key is present in dict 1
        merged_dict[key] += value  #add it to the existing value
    else:
        merged_dict[key] = value #add the key value pair to the new dict
print(merged_dict)

#Exercise 2: Word Frequency Counter
#Given a list of words:
#words = ["apple", "banana", "apple", "orange", "banana", "apple"]

#Create a dictionary word_counts that stores how many times each word appears.

#Example Output: {"apple": 3, "banana": 2, "orange": 1}
words = ["apple","banana","apple","orange","banana","apple"]
word_freq_counter ={}

for word in words: #word is temp variable for each fruit in the list
    if word in word_freq_counter:
        word_freq_counter[word] +=1
    else:
        word_freq_counter[word] =1
print (word_freq_counter)

#Exercise 3: Filter Dictionary by Value

#Method 1

#Given a dictionary of students and their marks:
#students = {"Alice": 85, "Bob": 72, "Charlie": 90, "David": 65}

#Create a new dictionary top_students with students who scored 75 or higher.
#Example Output: {"Alice": 85, "Charlie": 90}

students ={"Alice":85,"Bob":72,"Charlie":90,"David":65}
top_students={}
for student,marks in students.items():
    if (marks>= 75):
        top_students[student]=marks
print(top_students)

#Method 2 for Exercise 3
students ={"Alice":85,"Bob":72,"Charlie":90,"David":65}
top_students = {student :marks for student, marks in students.items() if score>=75}
print(top_students)



