from math import factorial

def Cn(n, x):
	return factorial(n) // (factorial(x) * factorial(n - x))

def binomial(n, p, x):
    return Cn(n,x) * p**n * (1-p)**(n-x)