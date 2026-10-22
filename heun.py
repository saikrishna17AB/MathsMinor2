import math
import os
import csv
import matplotlib.pyplot as plt

# ---------------- problem definition ----------------
# dy/dx = x^2 + y,   y(0) = 1

def ode_function(x, y):
    """Right hand side f(x, y) of the differential equation."""
    return x * x + y

def exact_solution(x):
    """Analytical solution, used to measure the error of the numerical method."""
    return 3 * math.exp(x) - x * x - 2 * x - 2

# ---------------- parameters ----------------
x_initial = 0
y_initial = 1
step_size = 0.05
x_final = 2

# number of iterations (40 steps, same as the original code)
number_of_steps = int(round((x_final - x_initial) / step_size))

# lists that store the numerical solution
x_values = [x_initial]
y_values = [y_initial]

# ---------------- numerical method: Heun's Method ----------------
# Heun (modified Euler): average of the slope at both ends of the interval
x_current = x_initial
y_current = y_initial

for step_index in range(number_of_steps):
    # slope at the start of the interval
    slope_start = ode_function(x_current, y_current)

    # predictor: plain Euler step to estimate y at the end of the interval
    y_predicted = y_current + step_size * slope_start

    x_next = x_initial + (step_index + 1) * step_size

    # slope at the end of the interval, using the predicted y
    slope_end = ode_function(x_next, y_predicted)

    # corrector: use the average of the two slopes
    y_current = y_current + (step_size / 2) * (slope_start + slope_end)

    x_current = x_next

    x_values.append(x_current)
    y_values.append(y_current)

# ---------------- exact solution (smooth curve for plotting) ----------------
x_exact_curve = []
y_exact_curve = []

for point_index in range(2001):
    x_point = point_index * 0.001
    x_exact_curve.append(x_point)
    y_exact_curve.append(exact_solution(x_point))

# ---------------- print comparison table ----------------
print("Heun's Method (order 2)")
print()

print("x\t\tHeun\t\tExact\t\tHeun Error")

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
print("Number of iterations:", len(x_values) - 1)

print()
print("Value at x =", x_final)
print()

print("Heun's Method:")
print(y_values[-1])

print()
print("Exact Solution:")
print(exact_solution(x_final))

print()
print("Heun's Method Absolute Error:")
print(abs(exact_solution(x_final) - y_values[-1]))

# ---------------- save the numerical values ----------------
# plot_all.py reads this file to draw every method in one graph.
# The results folder is created here, when the script is executed.
os.makedirs("results", exist_ok=True)

with open("results/heun.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["x", "y"])
    for row_index in range(len(x_values)):
        writer.writerow([x_values[row_index], y_values[row_index]])

# ---------------- plot ----------------
plt.figure(figsize=(10, 6))

plt.plot(x_exact_curve, y_exact_curve, label="Exact Solution")
plt.plot(x_values, y_values, marker="o", markersize=3, label="Heun's Method")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Heun's Method vs Exact Solution")
plt.legend()
plt.grid(True)

plt.savefig("results/heun.png", dpi=300, bbox_inches="tight")

plt.show()
