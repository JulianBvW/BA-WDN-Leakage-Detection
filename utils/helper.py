def any_transform(a, b):
    any = lambda l: int(sum(l) > 0)
    
    a = list(map(any, a))
    b = list(map(any, b))

    return a, b