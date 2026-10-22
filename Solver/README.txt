ODE NUMERICAL SOLVER
====================

Solves   dy/dx = x^2 + y   with the numerical method(s) you choose.

How to run
----------
Put the three files together in one folder (problem.py, methods.py, main.py) and run:

    python main.py

You will be asked for:
  1. the method(s): a number (4), several numbers (1,4,13), or "all"
  2. the initial point x0 and y(x0)
  3. the target x value
  4. the step size h
  5. the number of Picard iterations (only if Picard is chosen)

Pressing Enter keeps the default (x0 = 0, y(0) = 1, target = 2, h = 0.05).

Output
------
The values at every step are printed on the screen and saved in the
"outputs" folder, in a separate sub folder for every run, for example

    outputs/rk4_x0=0_y0=1_target=2_h=0.05/
        rk4_values.csv     step, x, numerical value, exact value, error
        rk4_graph.png      solution vs exact curve, and the error graph
        summary.txt        parameters and final values

When more than one method is chosen you also get

        comparison_graph.png      all methods and the exact solution
        comparison_errors.png     error of every method (log scale)
        comparison_values.csv     all methods side by side
        and a ranking (smallest error first) printed and saved in summary.txt

Notes
-----
* If the target is not an exact multiple of h, h is adjusted slightly so that
  a whole number of steps ends exactly at the target.
* Multistep methods (Adams-Bashforth, Adams-Moulton, Milne-Simpson) get their
  starting values from the 4th order Taylor series. The implicit formulas use
  a Taylor series predictor and an iterated corrector.
* The exact solution is recomputed for whatever starting point you enter.
* The original method scripts are not used or changed by this program.
