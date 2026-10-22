import math
import os
import csv
import matplotlib.pyplot as plt

# ---------------- problem definition ----------------
# dy/dx = x^2 + y,   y(0) = 1

def exact_solution(x):
    """Analytical solution, used to measure the error of the numerical method."""
    return 3 * math.exp(x) - x * x - 2 * x - 2

# ---------------- parameters ----------------
x_initial = 0
y_initial = 1
step_size = 0.05
x_final = 2

# number of points where the solution is evaluated (40 steps, same as the original code)
number_of_steps = int(round((x_final - x_initial) / step_size))

# Picard's method is not a step by step method. It builds a polynomial in x,
# and every Picard iteration makes that polynomial more accurate.
number_of_picard_iterations = 10

# ---------------- Picard's method ----------------
# y_(k+1)(x) = y(0) + integral from 0 to x of f(t, y_k(t)) dt,   with f(t, y) = t^2 + y
#
# The polynomial is stored as a list of coefficients:
# coefficients[k] is the coefficient of x^k, so [1, 1, 0, 1/3] means 1 + x + x^3/3.

# y_0(x) = y(0), a constant polynomial
coefficients = [y_initial]

print("Picard polynomials")
print()
print("y_0(x) =", coefficients)

for iteration in range(1, number_of_picard_iterations + 1):
    # Step 1: build f(t, y_k(t)) = t^2 + y_k(t)
    # (copy the coefficients of y_k and add 1 to the coefficient of t^2)
    integrand = list(coefficients)
    while len(integrand) < 3:
        integrand.append(0)
    integrand[2] = integrand[2] + 1

    # Step 2: integrate term by term from 0 to x
    # the integral of c * t^k is c / (k + 1) * x^(k+1), so the polynomial moves up one degree
    integrated = [0]
    for power in range(len(integrand)):
        integrated.append(integrand[power] / (power + 1))

    # Step 3: add the initial value y(0) (the constant term)
    integrated[0] = integrated[0] + y_initial

    coefficients = integrated

    print(f"After Picard iteration {iteration}, highest power of x = {len(coefficients) - 1}")

print()
print("Final coefficients of y(x) = c0 + c1*x + c2*x^2 + ...")
for power in range(len(coefficients)):
    print(f"c{power} = {coefficients[power]:.12f}")

def picard_value(x):
    """Evaluate the final Picard polynomial at x."""
    value = 0
    for power in range(len(coefficients)):
        value = value + coefficients[power] * x ** power
    return value

# ---------------- evaluate the polynomial at the same points as the other methods ----------------
x_values = []
y_values = []

for step_index in range(number_of_steps + 1):
    x_point = x_initial + step_index * step_size
    x_values.append(x_point)
    y_values.append(picard_value(x_point))

# ---------------- exact solution (smooth curve for plotting) ----------------
x_exact_curve = []
y_exact_curve = []

for point_index in range(2001):
    x_point = point_index * 0.001
    x_exact_curve.append(x_point)
    y_exact_curve.append(exact_solution(x_point))

# ---------------- print comparison table ----------------
print()
print("Picard Method")
print()

print("x\t\tPicard\t\tExact\t\tPicard Error")

for row_index in range(len(x_values)):
    x_point = x_values[row_index]

    numerical_value = y_values[row_index]
    exact_value = exact_solution(x_point)

    absolute_error = abs(exact_value - numerical_value)

    print(
        f"{x_point:.2f}\t\t"
        f"{numerical_value:.9f}\t"
        f"{exact_value:.9f}\t"
        f"{absolute_error:.9f}"
    )

print()
print("Number of Picard iterations:", number_of_picard_iterations)
print("Number of points evaluated:", len(x_values) - 1)

print()
print("Value at x =", x_final)
print()

print("Picard Method:")
print(y_values[-1])

print()
print("Exact Solution:")
print(exact_solution(x_final))

print()
print("Picard Method Absolute Error:")
print(abs(exact_solution(x_final) - y_values[-1]))

# ---------------- save the numerical values ----------------
# plot_all.py reads this file to draw every method in one graph.
# The results folder is created here, when the script is executed.
os.makedirs("results", exist_ok=True)

with open("results/picard.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["x", "y"])
    for row_index in range(len(x_values)):
        writer.writerow([x_values[row_index], y_values[row_index]])

# ---------------- plot ----------------
plt.figure(figsize=(10, 6))

plt.plot(x_exact_curve, y_exact_curve, label="Exact Solution")
plt.plot(x_values, y_values, marker="o", markersize=3, label="Picard Method")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Picard Method vs Exact Solution")
plt.legend()
plt.grid(True)

plt.savefig("results/picard.png", dpi=300, bbox_inches="tight")

plt.show()