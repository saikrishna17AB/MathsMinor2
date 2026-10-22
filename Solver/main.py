import os
import csv
import matplotlib.pyplot as plt

from problem import exact_solution
from methods import METHOD_LIST

# All results are saved in the "outputs" folder next to this file.
# Every run gets its own sub folder named after the method and the parameters.
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


# ---------------------------------------------------------------------
#                            reading the input
# ---------------------------------------------------------------------

def ask_number(question, default, number_type=float, minimum=None):
    """Ask for a number. Pressing Enter gives the default value."""
    while True:
        answer = input(f"{question} [{default}]: ").strip()

        if answer == "":
            return default

        try:
            value = number_type(answer)
        except ValueError:
            print("  Please enter a valid number.")
            continue

        if minimum is not None and value < minimum:
            print(f"  The value must be at least {minimum}.")
            continue

        return value


def choose_methods():
    """Show the menu and return the list of chosen methods."""
    print()
    print("Available methods:")
    for position, (key, name, function) in enumerate(METHOD_LIST, start=1):
        print(f"  {position:2d}. {name}")
    print(f"  {len(METHOD_LIST) + 1:2d}. All methods (with comparison)")
    print()

    while True:
        answer = input("Choose method number(s), e.g. 4  or  1,4,13  or  all: ").strip().lower()

        if answer in ("all", str(len(METHOD_LIST) + 1)):
            return list(METHOD_LIST)

        try:
            numbers = [int(part) for part in answer.split(",")]
        except ValueError:
            print("  Please type numbers separated by commas, or 'all'.")
            continue

        if all(1 <= number <= len(METHOD_LIST) for number in numbers):
            # remove repeated numbers but keep the order
            chosen = []
            for number in numbers:
                if METHOD_LIST[number - 1] not in chosen:
                    chosen.append(METHOD_LIST[number - 1])
            return chosen

        print(f"  Numbers must be between 1 and {len(METHOD_LIST)}.")


# ---------------------------------------------------------------------
#                       running a method and showing it
# ---------------------------------------------------------------------

def run_method(key, name, function, x_initial, y_initial, step_size, number_of_steps, picard_iterations):
    """Run one method and collect its values and errors."""
    if key == "picard":
        x_values, y_values = function(x_initial, y_initial, step_size, number_of_steps, picard_iterations)
    else:
        x_values, y_values = function(x_initial, y_initial, step_size, number_of_steps)

    exact_values = [exact_solution(x, x_initial, y_initial) for x in x_values]
    errors = [abs(exact_values[i] - y_values[i]) for i in range(len(x_values))]

    return {
        "key": key,
        "name": name,
        "x_values": x_values,
        "y_values": y_values,
        "exact_values": exact_values,
        "errors": errors,
    }


def print_table(result):
    """Print the value of the method, the exact value and the error at every step."""
    print()
    print("=" * 78)
    print(result["name"])
    print("=" * 78)
    print(f"{'step':>4s}  {'x':>10s}  {'Numerical':>16s}  {'Exact':>16s}  {'Absolute error':>16s}")

    for index in range(len(result["x_values"])):
        print(
            f"{index:4d}  "
            f"{result['x_values'][index]:10.4f}  "
            f"{result['y_values'][index]:16.9f}  "
            f"{result['exact_values'][index]:16.9f}  "
            f"{result['errors'][index]:16.9f}"
        )

    print()
    print(f"Value at x = {result['x_values'][-1]:.6g}:")
    print(f"  {result['name']}: {result['y_values'][-1]:.12f}")
    print(f"  Exact solution: {result['exact_values'][-1]:.12f}")
    print(f"  Absolute error: {result['errors'][-1]:.9e}")


# ---------------------------------------------------------------------
#                              saving the results
# ---------------------------------------------------------------------

def save_table(result, folder):
    """Save the step by step values of one method as a csv file."""
    file_path = os.path.join(folder, result["key"] + "_values.csv")

    with open(file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "x", "numerical", "exact", "absolute_error"])
        for index in range(len(result["x_values"])):
            writer.writerow([
                index,
                result["x_values"][index],
                result["y_values"][index],
                result["exact_values"][index],
                result["errors"][index],
            ])


def exact_curve(x_initial, x_target, y_initial):
    """Many closely spaced points of the exact solution, for a smooth curve."""
    points = 400
    x_curve = [x_initial + (x_target - x_initial) * i / points for i in range(points + 1)]
    y_curve = [exact_solution(x, x_initial, y_initial) for x in x_curve]
    return x_curve, y_curve


def plot_single_method(result, x_initial, y_initial, x_target, folder):
    """Graph of one method: solution against the exact curve, and its error."""
    x_curve, y_curve = exact_curve(x_initial, x_target, y_initial)

    figure, (solution_axes, error_axes) = plt.subplots(1, 2, figsize=(14, 6))

    solution_axes.plot(x_curve, y_curve, label="Exact Solution")
    solution_axes.plot(result["x_values"], result["y_values"], marker="o", markersize=3, label=result["name"])
    solution_axes.set_xlabel("x")
    solution_axes.set_ylabel("y")
    solution_axes.set_title(result["name"] + " vs Exact Solution")
    solution_axes.legend()
    solution_axes.grid(True)

    error_axes.plot(result["x_values"], result["errors"], marker="o", markersize=3, color="tab:red")
    error_axes.set_xlabel("x")
    error_axes.set_ylabel("Absolute error")
    error_axes.set_title("Error of " + result["name"])
    error_axes.grid(True)

    figure.savefig(os.path.join(folder, result["key"] + "_graph.png"), dpi=300, bbox_inches="tight")

    return figure


def plot_comparison(results, x_initial, y_initial, x_target, folder):
    """One graph with all chosen methods, and one graph with all their errors."""
    x_curve, y_curve = exact_curve(x_initial, x_target, y_initial)

    # --- all methods and the exact solution ---
    comparison_figure = plt.figure(figsize=(12, 7))
    plt.plot(x_curve, y_curve, color="black", linewidth=2.5, label="Exact Solution")

    for result in results:
        plt.plot(result["x_values"], result["y_values"], marker="o", markersize=2, linewidth=1, label=result["name"])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Chosen Methods vs Exact Solution")
    plt.legend()
    plt.grid(True)
    comparison_figure.savefig(os.path.join(folder, "comparison_graph.png"), dpi=300, bbox_inches="tight")

    # --- error of every method (log scale) ---
    error_figure = plt.figure(figsize=(12, 7))

    for result in results:
        # tiny floor so that an exact zero error can be drawn on a log axis
        floored_errors = [max(error, 1e-16) for error in result["errors"]]
        plt.semilogy(result["x_values"], floored_errors, label=result["name"])

    plt.xlabel("x")
    plt.ylabel("Absolute error (log scale)")
    plt.title("Error of Each Method")
    plt.legend()
    plt.grid(True)
    error_figure.savefig(os.path.join(folder, "comparison_errors.png"), dpi=300, bbox_inches="tight")

    return comparison_figure, error_figure


def save_comparison_table(results, folder):
    """csv with the value and the error of every method at every x."""
    file_path = os.path.join(folder, "comparison_values.csv")

    with open(file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        header = ["x", "exact"]
        for result in results:
            header.append(result["name"])
            header.append(result["name"] + " error")
        writer.writerow(header)

        for index in range(len(results[0]["x_values"])):
            row = [results[0]["x_values"][index], results[0]["exact_values"][index]]
            for result in results:
                row.append(result["y_values"][index])
                row.append(result["errors"][index])
            writer.writerow(row)


def rank_methods(results):
    """Print the ranking of the methods (smallest error first). Returns the text."""
    lines = []
    lines.append(f"{'Method':38s}{'Error at last x':>20s}{'Maximum error':>20s}")

    by_final_error = sorted(results, key=lambda result: result["errors"][-1])

    for position, result in enumerate(by_final_error, start=1):
        lines.append(
            f"{position:2d}. {result['name']:34s}"
            f"{result['errors'][-1]:20.9e}"
            f"{max(result['errors']):20.9e}"
        )

    best_final = by_final_error[0]
    best_maximum = min(results, key=lambda result: max(result["errors"]))

    lines.append("")
    lines.append(f"Lowest error at the last x: {best_final['name']} ({best_final['errors'][-1]:.9e})")
    lines.append(f"Lowest maximum error:       {best_maximum['name']} ({max(best_maximum['errors']):.9e})")

    return "\n".join(lines)


def save_summary(results, parameters_text, ranking_text, folder):
    """Text file with the parameters used and the final values."""
    with open(os.path.join(folder, "summary.txt"), "w") as summary_file:
        summary_file.write("dy/dx = x^2 + y\n")
        summary_file.write(parameters_text + "\n\n")

        for result in results:
            summary_file.write(
                f"{result['name']}: value at last x = {result['y_values'][-1]:.12f}, "
                f"exact = {result['exact_values'][-1]:.12f}, "
                f"error = {result['errors'][-1]:.9e}\n"
            )

        if ranking_text:
            summary_file.write("\nRanking\n")
            summary_file.write(ranking_text + "\n")


def make_run_folder(chosen_methods, x_initial, y_initial, x_target, step_size):
    """Create (if needed) and return the folder where this run is saved."""
    if len(chosen_methods) == 1:
        method_part = chosen_methods[0][0]
    else:
        method_part = "compare_" + str(len(chosen_methods)) + "_methods"

    folder_name = f"{method_part}_x0={x_initial:g}_y0={y_initial:g}_target={x_target:g}_h={step_size:g}"

    # keep only characters that are safe in a file name
    folder_name = "".join(c if (c.isalnum() or c in "._=-") else "_" for c in folder_name)

    folder = os.path.join(OUTPUT_ROOT, folder_name)
    os.makedirs(folder, exist_ok=True)

    return folder


# ---------------------------------------------------------------------
#                                 main program
# ---------------------------------------------------------------------

def solve_one_problem():
    print()
    print("Solving dy/dx = x^2 + y  numerically")

    chosen_methods = choose_methods()

    print()
    print("Enter the problem data (press Enter to accept the value in brackets).")
    x_initial = ask_number("Initial x (x0)", 0.0)
    y_initial = ask_number("Initial y, y(x0)", 1.0)

    while True:
        x_target = ask_number("Target x value (where you want the solution)", 2.0)
        if x_target > x_initial:
            break
        print("  The target must be greater than the initial x.")

    step_size = ask_number("Step size h", 0.05)
    while step_size <= 0:
        print("  The step size must be positive.")
        step_size = ask_number("Step size h", 0.05)

    picard_iterations = 10
    if any(key == "picard" for key, name, function in chosen_methods):
        picard_iterations = ask_number("Number of Picard iterations", 10, int, minimum=1)

    # number of steps needed to go from x0 to the target
    span = x_target - x_initial
    number_of_steps = max(1, int(round(span / step_size)))
    used_step_size = span / number_of_steps

    if abs(used_step_size - step_size) > 1e-12:
        print()
        print(f"Note: the target is not an exact multiple of h = {step_size:g}.")
        print(f"      h is adjusted to {used_step_size:.10g} so that exactly {number_of_steps} steps reach x = {x_target:g}.")

    step_size = used_step_size

    parameters_text = (
        f"x0 = {x_initial:g}, y(x0) = {y_initial:g}, target x = {x_target:g}, "
        f"h = {step_size:.10g}, number of steps = {number_of_steps}"
    )
    if any(key == "picard" for key, name, function in chosen_methods):
        parameters_text += f", Picard iterations = {picard_iterations}"

    print()
    print(parameters_text)

    folder = make_run_folder(chosen_methods, x_initial, y_initial, x_target, step_size)

    # run every chosen method, print its table and save its files
    results = []
    figures = []

    for key, name, function in chosen_methods:
        result = run_method(key, name, function, x_initial, y_initial, step_size, number_of_steps, picard_iterations)
        results.append(result)

        print_table(result)
        save_table(result, folder)

        figure = plot_single_method(result, x_initial, y_initial, x_target, folder)

        if len(chosen_methods) == 1:
            figures.append(figure)      # shown on the screen
        else:
            plt.close(figure)           # saved only, to avoid dozens of windows

    # comparison of several methods
    ranking_text = ""

    if len(results) > 1:
        ranking_text = rank_methods(results)

        print()
        print("=" * 78)
        print("Comparison of the chosen methods")
        print("=" * 78)
        print(ranking_text)

        save_comparison_table(results, folder)
        figures.extend(plot_comparison(results, x_initial, y_initial, x_target, folder))

    save_summary(results, parameters_text, ranking_text, folder)

    print()
    print("Results (tables, graphs and summary) were saved in:")
    print("  " + folder)

    # show the graphs on the screen (close the windows to continue)
    if figures:
        plt.show()

    for figure in figures:
        plt.close(figure)


def main():
    print("=" * 78)
    print("                      ODE NUMERICAL SOLVER")
    print("=" * 78)

    while True:
        solve_one_problem()

        print()
        again = input("Solve another problem? (y/n) [n]: ").strip().lower()
        if again not in ("y", "yes"):
            break

    print("Done.")


if __name__ == "__main__":
    main()
