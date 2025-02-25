#accept user input of goal and a deadline
#print back how much time is left until that deadline

#we will be using a datetime module for the time calculation

import datetime

user_input = input("enter your goal with a deadline seperated by colon\n")
input_list = user_input.split(":") #spliting to access
goal=input_list[0]
deadline=input_list[1]
deadline_date =datetime.datetime.strptime(deadline,"%d.%m.%Y")

today_date=datetime.datetime.today() #using the fn today in datetime

#calculate how ,many days monthss and years are left
time_left=deadline_date-today_date
print(f"dear user, time left for {goal} is {time_left}")
print(f"dear user, time left for {goal} is {time_left.days}") #.days to print just days not time
print(f"dear user, time left for {goal} is {time_left.total_seconds()}") #for seconds
print(f"dear user, time left for {goal} is {time_left.total_seconds()/60/60}") #for hours

"""print(datetime.datetime.strptime(deadline,"%d.%m.%Y"))

print(type(datetime.datetime.strptime(deadline,"%d.%m.%Y")))"""




