def arithmetic(x, y):
    if isinstance(y, str):
        if isinstance(x, str):
            return x + y
        else:  # isinstance(x, float)
            return str(x) + y
    else:  # isinstance(y, float)
        if isinstance(x, str):
            return x * int(y)
        else:  # isinstance(x, float)
            return x * y
print(arithmetic("Hello", "World"))   
print(arithmetic(4.1, "Test"))        
print(arithmetic("Hi", 5.5))         
print(arithmetic(9.5, 2.0))