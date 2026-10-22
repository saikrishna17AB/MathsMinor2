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

def taylor_step(x_current, y_current, step_size):
    """One step of the 4th order Taylor series for y' = x^2 + y."""
    # Derivatives of y obtained by differentiating the ODE repeatedly
    first_derivative = x_current * x_current + y_current      # y'   = x^2 + y
    second_derivative = 2 * x_current + first_derivative      # y''  = 2x + y'
    third_derivative = 2 + second_derivative                  # 3rd derivative = 2 + y''
    fourth_derivative = third_derivative                      # 4th derivative = 3rd derivative

    # y(x+h) = y + h*y' + h^2/2! * y'' + h^3/3! * y(3) + h^4/4! * y(4)
    return (y_current
            + step_size * first_derivative
            + (step_size ** 2 / 2) * second_derivative
            + (step_size ** 3 / 6) * third_derivative
            + (step_size ** 4 / 24) * fourth_derivative)

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

# ---------------- numerical method: Adams-Bashforth 4-step ----------------
# A multistep method needs several starting points but only y(0) is given,
# so the missing starting values are obtained with the 4th order Taylor series.
for step_index in range(3):
    y_start = taylor_step(x_values[step_index], y_values[step_index], step_size)
    x_values.append(x_initial + (step_index + 1) * step_size)
    y_values.append(y_start)

# Explicit multistep loop: step_index is the latest known point,
# and the point step_index + 1 is computed from the previous slopes.
for step_index in range(3, number_of_steps):
    x_next = x_initial + (step_index + 1) * step_size

    # slopes at the current and the three previous points
    slope_current = ode_function(x_values[step_index], y_values[step_index])
    slope_previous = ode_function(x_values[step_index - 1], y_values[step_index - 1])
    slope_two_back = ode_function(x_values[step_index - 2], y_values[step_index - 2])
    slope_three_back = ode_function(x_values[step_index - 3], y_values[step_index - 3])

    # y_next = y_n + h/24 * (55 f_n - 59 f_(n-1) + 37 f_(n-2) - 9 f_(n-3))
    y_next = y_values[step_index] + (step_size / 24) * (
        55 * slope_current - 59 * slope_previous + 37 * slope_two_back - 9 * slope_three_back)

    x_values.append(x_next)
    y_values.append(y_next)

# ---------------- exact solution (smooth curve for plotting) ----------------
x_exact_curve = []
y_exact_curve = []

for point_index in range(2001):
    x_point = point_index * 0.001
    x_exact_curve.append(x_point)
    y_exact_curve.append(exact_solution(x_point))

# ---------------- print comparison table ----------------
print("Adams-Bashforth 4-step (order 4)")
print()

print("x\t\tAB4\t\tExact\t\tAB4 Error")

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

print("Adams-Bashforth 4-step:")
print(y_values[-1])

print()
print("Exact Solution:")
print(exact_solution(x_final))

print()
print("Adams-Bashforth 4-step Absolute Error:")
print(abs(exact_solution(x_final) - y_values[-1]))

# ---------------- save the numerical values ----------------
# plot_all.py reads this file to draw every method in one graph.
# The results folder is created here, when the script is executed.
os.makedirs("results", exist_ok=True)

with open("results/ab4.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["x", "y"])
    for row_index in range(len(x_values)):
        writer.writerow([x_values[row_index], y_values[row_index]])

# ---------------- plot ----------------
plt.figure(figsize=(10, 6))

plt.plot(x_exact_curve, y_exact_curve, label="Exact Solution")
plt.plot(x_values, y_values, marker="o", markersize=3, label="Adams-Bashforth 4-step")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Adams-Bashforth 4-step vs Exact Solution")
plt.legend()
plt.grid(True)

plt.savefig("results/ab4.png", dpi=300, bbox_inches="tight")

plt.show()
