def my_function(**kwargs):
    for key, value in kwargs.items():
        print(f"Argument {key} has value {value}")

# Call the function with arbitrary keyword arguments
my_function(arg1="value1", arg2="value2", arg3="value3")