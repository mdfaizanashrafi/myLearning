try:
    my_list = ["January", "February", "March"]
    print(my_list)
    my_list.append("April") #add to list
    print(my_list)
except ValueError:
    print("Invalid input,please enter int")
