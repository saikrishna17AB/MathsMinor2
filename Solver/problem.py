import math

# The differential equation solved by every method:
#
#       dy/dx = x^2 + y
#
# Its general solution is  y(x) = C * e^x - x^2 - 2x - 2,
# where the constant C depends on the starting point (x0, y0).

def ode_function(x, y):
    """Right hand side f(x, y) of the differential equation."""
    return x * x + y


def exact_solution(x, x_initial, y_initial):
    """Analytical solution passing through the point (x_initial, y_initial).

    Used to measure the error of every numerical method.
    For x_initial = 0 and y_initial = 1 this is 3*e^x - x^2 - 2x - 2.
    """
    # Choose C so that the curve passes through the starting point
    constant = (y_initial + x_initial ** 2 + 2 * x_initial + 2) * math.exp(-x_initial)

    return constant * math.exp(x) - x * x - 2 * x - 2
