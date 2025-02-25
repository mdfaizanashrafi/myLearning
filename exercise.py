calculation_to_units = 24
name_of_unit = "hours"

def days_to_units(number_of_days):
    return (f"{number_of_days} hours in day {number_of_days* calculation_to_units} {name_of_unit}")

def validate_and_execute():
    try:
        user_input_int = int(number_of_days)
        if user_input_int>0:
            calculated_value = days_to_units(user_input_int)
            print(calculated_value)
        elif user_input_int==0:
            print("yu entered 0")
    except ValueError:
        print("Invalid input,please enter int")
