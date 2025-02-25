import exercise

user_input =""
while user_input != "exit":
    user_input = input("Hey Enter the dat\n")
    list_of_days = user_input.split()
    print(list_of_days)
    print(set(list_of_days))
    print(type(list_of_days))
    print(type(set(list_of_days)))
    for num_of_days in set(user_input.split()):
        exercise.validate_and_execute()
