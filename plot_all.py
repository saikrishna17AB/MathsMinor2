import math
import csv
import os
import matplotlib.pyplot as plt

# Run the individual method scripts first (euler.py, heun.py, ...).
# Each of them saves its numerical solution to results/<method>.csv,
# which this script reads to compare all the methods together.

def exact_solution(x):
    """Analytical solution of dy/dx = x^2 + y, y(0) = 1."""
    return 3 * math.exp(x) - x * x - 2 * x - 2

x_final = 2

# (file name, name shown in the graph and the tables)
method_list = [
    ("euler", "Euler"),
    ("heun", "Heun"),
    ("midpoint", "Midpoint"),
    ("rk4", "Runge-Kutta 4"),
    ("taylor4", "Taylor 4"),
    ("ab2", "Adams-Bashforth 2"),
    ("ab3", "Adams-Bashforth 3"),
    ("ab4", "Adams-Bashforth 4"),
    ("am1", "Adams-Moulton 1 (Trapezoidal)"),
    ("am2", "Adams-Moulton 2"),
    ("am3", "Adams-Moulton 3"),
    ("am4", "Adams-Moulton 4"),
    ("milne_simpson", "Milne-Simpson"),
        ("picard", "Picard"),
]

# ---------------- exact solution (smooth curve) ----------------
x_exact_curve = []
y_exact_curve = []

for point_index in range(2001):
    x_point = point_index * 0.001
    x_exact_curve.append(x_point)
    y_exact_curve.append(exact_solution(x_point))

# ---------------- read the results of every method ----------------
# method_results maps file name -> (display name, x values, y values)
method_results = {}

for file_name, display_name in method_list:
    file_path = "results/" + file_name + ".csv"

    if not os.path.exists(file_path):
        print("Missing", file_path, "- run", file_name + ".py first")
        continue

    x_values = []
    y_values = []

    with open(file_path) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # skip the header row
        for row in reader:
            x_values.append(float(row[0]))
            y_values.append(float(row[1]))

    method_results[file_name] = (display_name, x_values, y_values)

if not method_results:
    raise SystemExit("No results found. Run the method scripts first.")

# ---------------- error comparison ----------------
print("Comparison of all methods")
print()
print(f"{'Method':32s}{'Value at x=2':>18s}{'Error at x=2':>18s}{'Max error':>18s}")

error_summary = []   # (display name, value at x=2, error at x=2, max error)

for file_name, (display_name, x_values, y_values) in method_results.items():
    # absolute error at every point of this method
    errors = [abs(exact_solution(x_values[i]) - y_values[i]) for i in range(len(x_values))]

    error_at_end = errors[-1]
    maximum_error = max(errors)

    error_summary.append((display_name, y_values[-1], error_at_end, maximum_error))

    print(f"{display_name:32s}{y_values[-1]:18.9f}{error_at_end:18.9e}{maximum_error:18.9e}")

print()
print("Exact value at x = 2:", exact_solution(x_final))

# rank the methods: smallest error at x = 2 comes first
print()
print("Ranking by error at x = 2 (smallest first):")

ranking = sorted(error_summary, key=lambda entry: entry[2])

for position, entry in enumerate(ranking, start=1):
    print(f"{position:2d}. {entry[0]:32s}{entry[2]:.9e}")

best_at_end = ranking[0]
best_overall = min(error_summary, key=lambda entry: entry[3])

print()
print("Lowest error at x = 2:", best_at_end[0], "(error =", best_at_end[2], ")")
print("Lowest maximum error over [0, 2]:", best_overall[0], "(error =", best_overall[3], ")")

# ---------------- graph 1: all methods and the exact solution ----------------
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(12, 7))

plt.plot(x_exact_curve, y_exact_curve, color="black", linewidth=2.5, label="Exact Solution")

for file_name, (display_name, x_values, y_values) in method_results.items():
    plt.plot(x_values, y_values, marker="o", markersize=2, linewidth=1, label=display_name)

plt.xlabel("x")
plt.ylabel("y")
plt.title("All Numerical Methods vs Exact Solution")
plt.legend()
plt.grid(True)

plt.savefig("results/all_methods_comparison.png", dpi=300, bbox_inches="tight")

# ---------------- graph 2: error of every method (log scale) ----------------
plt.figure(figsize=(12, 7))

for file_name, (display_name, x_values, y_values) in method_results.items():
    # tiny floor (1e-16) so that an exact zero error at x = 0 can be drawn on a log axis
    errors = [max(abs(exact_solution(x_values[i]) - y_values[i]), 1e-16) for i in range(len(x_values))]
    plt.semilogy(x_values, errors, label=display_name)

plt.xlabel("x")
plt.ylabel("Absolute error (log scale)")
plt.title("Error of Each Method")
plt.legend()
plt.grid(True)

plt.savefig("results/all_methods_error.png", dpi=300, bbox_inches="tight")

plt.show()
