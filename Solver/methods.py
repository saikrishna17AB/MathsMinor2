from problem import ode_function

# Every method has the same inputs and the same outputs:
#
#   inputs : x_initial, y_initial, step_size, number_of_steps
#   outputs: x_values, y_values   (lists, starting with the initial point)


def make_x_values(x_initial, step_size, number_of_steps):
    """x points of the grid. Computed from the index (not by repeated addition)
    so that floating point errors do not build up."""
    return [x_initial + step_index * step_size for step_index in range(number_of_steps + 1)]


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


def add_taylor_starting_values(x_values, y_values, step_size, number_of_starting_values):
    """A multistep method needs several starting points but only y(x0) is known,
    so the missing starting values are computed with the 4th order Taylor series."""
    number_of_steps = len(x_values) - 1

    for step_index in range(min(number_of_starting_values, number_of_steps)):
        y_start = taylor_step(x_values[step_index], y_values[step_index], step_size)
        y_values.append(y_start)


def iterate_corrector(corrector_formula, y_predicted):
    """Solve an implicit corrector formula by repeating it until the value stops changing.

    corrector_formula(y_guess) returns the right hand side of the corrector
    evaluated with y_guess in place of the unknown y_next.
    """
    y_corrected = y_predicted

    for iteration in range(50):
        y_updated = corrector_formula(y_corrected)
        has_converged = abs(y_updated - y_corrected) < 1e-13
        y_corrected = y_updated
        if has_converged:
            break

    return y_corrected


# =====================================================================
#                          ONE-STEP METHODS
# =====================================================================

def euler_method(x_initial, y_initial, step_size, number_of_steps):
    # Euler: y_next = y_current + h * f(x_current, y_current)
    x_values = make_x_values(x_initial, step_size, number_of_steps)
    y_values = [y_initial]

    for step_index in range(number_of_steps):
        x_current = x_values[step_index]
        y_current = y_values[step_index]

        # slope at the start of the interval is used for the whole step
        y_next = y_current + step_size * ode_function(x_current, y_current)
        y_values.append(y_next)

    return x_values, y_values


def heun_method(x_initial, y_initial, step_size, number_of_steps):
    # Heun (modified Euler): average of the slope at both ends of the interval
    x_values = make_x_values(x_initial, step_size, number_of_steps)
    y_values = [y_initial]

    for step_index in range(number_of_steps):
        x_current = x_values[step_index]
        y_current = y_values[step_index]
        x_next = x_values[step_index + 1]

        # slope at the start of the interval
        slope_start = ode_function(x_current, y_current)

        # predictor: plain Euler step to estimate y at the end of the interval
        y_predicted = y_current + step_size * slope_start

        # slope at the end of the interval, using the predicted y
        slope_end = ode_function(x_next, y_predicted)

        # corrector: use the average of the two slopes
        y_next = y_current + (step_size / 2) * (slope_start + slope_end)
        y_values.append(y_next)

    return x_values, y_values


def midpoint_method(x_initial, y_initial, step_size, number_of_steps):
    # Midpoint: use the slope at the middle of the interval
    x_values = make_x_values(x_initial, step_size, number_of_steps)
    y_values = [y_initial]

    for step_index in range(number_of_steps):
        x_current = x_values[step_index]
        y_current = y_values[step_index]

        # slope at the start of the interval
        slope_start = ode_function(x_current, y_current)

        # half Euler step to estimate y at the midpoint of the interval
        y_midpoint = y_current + (step_size / 2) * slope_start

        # slope at the midpoint
        slope_midpoint = ode_function(x_current + step_size / 2, y_midpoint)

        # full step using the midpoint slope
        y_next = y_current + step_size * slope_midpoint
        y_values.append(y_next)

    return x_values, y_values


def runge_kutta_4_method(x_initial, y_initial, step_size, number_of_steps):
    # Classical fourth order Runge-Kutta
    x_values = make_x_values(x_initial, step_size, number_of_steps)
    y_values = [y_initial]

    for step_index in range(number_of_steps):
        x_current = x_values[step_index]
        y_current = y_values[step_index]

        # four slope estimates inside the interval
        slope_1 = ode_function(x_current, y_current)
        slope_2 = ode_function(x_current + step_size / 2, y_current + (step_size / 2) * slope_1)
        slope_3 = ode_function(x_current + step_size / 2, y_current + (step_size / 2) * slope_2)
        slope_4 = ode_function(x_current + step_size, y_current + step_size * slope_3)

        # weighted average of the slopes (weights 1, 2, 2, 1)
        y_next = y_current + (step_size / 6) * (slope_1 + 2 * slope_2 + 2 * slope_3 + slope_4)
        y_values.append(y_next)

    return x_values, y_values


def taylor_4_method(x_initial, y_initial, step_size, number_of_steps):
    # Fourth order Taylor series method (see taylor_step above)
    x_values = make_x_values(x_initial, step_size, number_of_steps)
    y_values = [y_initial]

    for step_index in range(number_of_steps):
        y_next = taylor_step(x_values[step_index], y_values[step_index], step_size)
        y_values.append(y_next)

    return x_values, y_values


# =====================================================================
#                   ADAMS-BASHFORTH (explicit multistep)
# =====================================================================
#
# y_next = y_n + h/denominator * (w0*f_n + w1*f_(n-1) + w2*f_(n-2) + ...)
#
# number of steps -> (denominator, [w0, w1, ...])
ADAMS_BASHFORTH_FORMULAS = {
    2: (2, [3, -1]),                  # two-step,   order 2
    3: (12, [23, -16, 5]),            # three-step, order 3
    4: (24, [55, -59, 37, -9]),       # four-step,  order 4
}


def adams_bashforth(number_of_points_used, x_initial, y_initial, step_size, number_of_steps):
    denominator, weights = ADAMS_BASHFORTH_FORMULAS[number_of_points_used]

    x_values = make_x_values(x_initial, step_size, number_of_steps)
    y_values = [y_initial]

    # starting values from the Taylor series
    add_taylor_starting_values(x_values, y_values, step_size, number_of_points_used - 1)

    # step_index is the latest known point; the point step_index + 1 is computed
    for step_index in range(number_of_points_used - 1, number_of_steps):
        weighted_slopes = 0

        # f_n, f_(n-1), f_(n-2), ... multiplied by their weights
        for points_back in range(number_of_points_used):
            slope = ode_function(x_values[step_index - points_back], y_values[step_index - points_back])
            weighted_slopes = weighted_slopes + weights[points_back] * slope

        y_next = y_values[step_index] + (step_size / denominator) * weighted_slopes
        y_values.append(y_next)

    return x_values, y_values


def adams_bashforth_2_method(x_initial, y_initial, step_size, number_of_steps):
    return adams_bashforth(2, x_initial, y_initial, step_size, number_of_steps)


def adams_bashforth_3_method(x_initial, y_initial, step_size, number_of_steps):
    return adams_bashforth(3, x_initial, y_initial, step_size, number_of_steps)


def adams_bashforth_4_method(x_initial, y_initial, step_size, number_of_steps):
    return adams_bashforth(4, x_initial, y_initial, step_size, number_of_steps)


# =====================================================================
#                   ADAMS-MOULTON (implicit multistep)
# =====================================================================
#
# y_next = y_n + h/denominator * (w0*f_(n+1) + w1*f_n + w2*f_(n-1) + ...)
# w0 multiplies the unknown f_(n+1), so the formula is implicit.
#
# number of known points -> (denominator, [w0, w1, ...])
ADAMS_MOULTON_FORMULAS = {
    1: (2, [1, 1]),                              # trapezoidal rule, order 2
    2: (12, [5, 8, -1]),                         # two-step,   order 3
    3: (24, [9, 19, -5, 1]),                     # three-step, order 4
    4: (720, [251, 646, -264, 106, -19]),        # four-step,  order 5
}


def adams_moulton(number_of_known_points, x_initial, y_initial, step_size, number_of_steps):
    denominator, weights = ADAMS_MOULTON_FORMULAS[number_of_known_points]

    x_values = make_x_values(x_initial, step_size, number_of_steps)
    y_values = [y_initial]

    # starting values from the Taylor series (the trapezoidal rule needs none)
    add_taylor_starting_values(x_values, y_values, step_size, number_of_known_points - 1)

    # step_index is the latest known point; the point step_index + 1 is computed
    for step_index in range(number_of_known_points - 1, number_of_steps):
        x_next = x_values[step_index + 1]

        # part of the formula that only uses known points: f_n, f_(n-1), ...
        known_slopes_part = 0
        for points_back in range(number_of_known_points):
            slope = ode_function(x_values[step_index - points_back], y_values[step_index - points_back])
            known_slopes_part = known_slopes_part + weights[points_back + 1] * slope

        # predictor: 4th order Taylor series step gives a first guess of y_next
        y_predicted = taylor_step(x_values[step_index], y_values[step_index], step_size)

        # corrector: the implicit formula, with y_guess in place of the unknown y_next
        def corrector_formula(y_guess):
            return y_values[step_index] + (step_size / denominator) * (
                weights[0] * ode_function(x_next, y_guess) + known_slopes_part)

        y_next = iterate_corrector(corrector_formula, y_predicted)
        y_values.append(y_next)

    return x_values, y_values


def adams_moulton_1_method(x_initial, y_initial, step_size, number_of_steps):
    return adams_moulton(1, x_initial, y_initial, step_size, number_of_steps)


def adams_moulton_2_method(x_initial, y_initial, step_size, number_of_steps):
    return adams_moulton(2, x_initial, y_initial, step_size, number_of_steps)


def adams_moulton_3_method(x_initial, y_initial, step_size, number_of_steps):
    return adams_moulton(3, x_initial, y_initial, step_size, number_of_steps)


def adams_moulton_4_method(x_initial, y_initial, step_size, number_of_steps):
    return adams_moulton(4, x_initial, y_initial, step_size, number_of_steps)


# =====================================================================
#                       MILNE-SIMPSON (predictor-corrector)
# =====================================================================

def milne_simpson_method(x_initial, y_initial, step_size, number_of_steps):
    x_values = make_x_values(x_initial, step_size, number_of_steps)
    y_values = [y_initial]

    # three starting values from the Taylor series
    add_taylor_starting_values(x_values, y_values, step_size, 3)

    # step_index is the latest known point; the point step_index + 1 is computed
    for step_index in range(3, number_of_steps):
        x_next = x_values[step_index + 1]

        # slopes at the known points
        slope_current = ode_function(x_values[step_index], y_values[step_index])
        slope_previous = ode_function(x_values[step_index - 1], y_values[step_index - 1])
        slope_two_back = ode_function(x_values[step_index - 2], y_values[step_index - 2])

        # Milne predictor (explicit): y_(n+1) = y_(n-3) + 4h/3 * (2 f_n - f_(n-1) + 2 f_(n-2))
        y_predicted = y_values[step_index - 3] + (4 * step_size / 3) * (
            2 * slope_current - slope_previous + 2 * slope_two_back)

        # Simpson corrector (implicit): y_(n+1) = y_(n-1) + h/3 * (f_(n+1) + 4 f_n + f_(n-1))
        def corrector_formula(y_guess):
            return y_values[step_index - 1] + (step_size / 3) * (
                ode_function(x_next, y_guess) + 4 * slope_current + slope_previous)

        y_next = iterate_corrector(corrector_formula, y_predicted)
        y_values.append(y_next)

    return x_values, y_values


# =====================================================================
#                              PICARD METHOD
# =====================================================================

def picard_method(x_initial, y_initial, step_size, number_of_steps, picard_iterations=10):
    # y_(k+1)(x) = y0 + integral from x0 to x of f(t, y_k(t)) dt,   with f(t, y) = t^2 + y
    #
    # The polynomial is written in the variable s = x - x0 and stored as a list of
    # coefficients: coefficients[k] multiplies s^k.
    coefficients = [y_initial]          # y_0(x) = y0, a constant

    for iteration in range(picard_iterations):
        # Step 1: build f(t, y_k(t)) = t^2 + y_k(t)
        integrand = list(coefficients)
        while len(integrand) < 3:
            integrand.append(0)

        # t^2 = (x0 + s)^2 = x0^2 + 2*x0*s + s^2
        integrand[0] = integrand[0] + x_initial ** 2
        integrand[1] = integrand[1] + 2 * x_initial
        integrand[2] = integrand[2] + 1

        # Step 2: integrate term by term; c * s^k becomes c/(k+1) * s^(k+1)
        # Step 3: the constant of integration is the initial value y0
        integrated = [y_initial]
        for power in range(len(integrand)):
            integrated.append(integrand[power] / (power + 1))

        coefficients = integrated

    # Evaluate the final polynomial at every grid point
    x_values = make_x_values(x_initial, step_size, number_of_steps)
    y_values = []

    for x_point in x_values:
        s = x_point - x_initial
        value = 0
        for power in range(len(coefficients)):
            value = value + coefficients[power] * s ** power
        y_values.append(value)

    return x_values, y_values


# =====================================================================
#                      LIST USED BY THE MENU IN main.py
# =====================================================================
# (key used in file names, name shown to the user, function)

METHOD_LIST = [
    ("euler", "Euler Method", euler_method),
    ("heun", "Heun's Method", heun_method),
    ("midpoint", "Midpoint Method", midpoint_method),
    ("rk4", "Runge-Kutta 4th Order", runge_kutta_4_method),
    ("taylor4", "Taylor Series 4th Order", taylor_4_method),
    ("ab2", "Adams-Bashforth 2-step", adams_bashforth_2_method),
    ("ab3", "Adams-Bashforth 3-step", adams_bashforth_3_method),
    ("ab4", "Adams-Bashforth 4-step", adams_bashforth_4_method),
    ("am1", "Adams-Moulton 1-step (Trapezoidal)", adams_moulton_1_method),
    ("am2", "Adams-Moulton 2-step", adams_moulton_2_method),
    ("am3", "Adams-Moulton 3-step", adams_moulton_3_method),
    ("am4", "Adams-Moulton 4-step", adams_moulton_4_method),
    ("milne_simpson", "Milne-Simpson Method", milne_simpson_method),
    ("picard", "Picard Method", picard_method),
]
