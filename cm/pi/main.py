from decimal import Decimal, getcontext

def calculate_pi(n_terms):
    getcontext().prec = 1000  # Set precision
    pi = Decimal(0)
    for k in range(n_terms):
        pi += Decimal((-1)**k) / Decimal(2*k + 1)
    pi *= 4
    return pi

n_terms = 1000000  # Number of terms in the series (increase for more precision)
pi_approximation = calculate_pi(n_terms)
print(pi_approximation)
