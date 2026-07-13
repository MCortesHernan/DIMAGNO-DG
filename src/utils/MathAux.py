'''
Auxiliary math functions
'''

def max(*args):
    '''
    Auxiliary function.
    Returns the maximum of two quantities a and b, without using branching.
    '''
    res = (args[0] + args[1] + abs(args[0]-args[1]))/2 
    for i in range(2, len(args)):
        res = (res + args[i] + abs(res-args[i]))/2 
    return res 

def maxabs(*args):
    '''
    Auxiliary function.
    Returns the maximum absolute value of two quantities a and b, without using branching.
    '''

    arg_abs = [*args]

    for i in range(len(arg_abs)):
        arg_abs[i] = abs(arg_abs[i])
    return max(*arg_abs)

def min(*args):
    '''
    Auxiliary function.
    Returns the minimum of two quantities a and b, without using branching.
    '''
    res = (args[0] + args[1] - abs(args[0]-args[1]))/2 
    for i in range(2, len(args)):
        res = (res + args[i] + abs(res-args[i]))/2 
    return res 


def minabs(*args):
    '''
    Auxiliary function.
    Returns the maximum absolute value of two quantities a and b, without using branching.
    '''

    arg_abs = [*args]

    for i in range(len(arg_abs)):
        arg_abs[i] = abs(arg_abs[i])

    return min(*arg_abs)

def maxminabs(*args):
    '''
    Auxiliary function.
    Returns the maximum absolute value of two quantities a and b, without using branching.
    '''

    arg_abs = [*args]

    for i in range(len(arg_abs)):
        arg_abs[i] = abs(arg_abs[i])

    return max(*arg_abs), min(*arg_abs)