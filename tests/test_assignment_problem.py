import pytest


# solve_assignment_problem

def test_solve_assignment_problem_square_max():
    """It solves a simple square maximization assignment correctly."""
    try:
        import numpy as np
        from blackboxphaseopt.assignment_problem import solve_assignment_problem

        weights = np.array([[1, 2],
                            [3, 4]])

        assignment, objective = solve_assignment_problem(weights, opt_type="max")

        assert assignment.shape == (2, 2)
        assert assignment.sum(axis=1).all() == 1
        assert assignment.sum(axis=0).all() == 1
        assert objective == 5

    except Exception as e:
        pytest.fail(f"solve_assignment_problem square max test failed: {e}")


def test_solve_assignment_problem_square_min():
    """It solves a simple square minimization assignment correctly."""
    try:
        import numpy as np
        from blackboxphaseopt.assignment_problem import solve_assignment_problem

        weights = np.array([[1, 2],
                            [3, 4]])

        assignment, objective = solve_assignment_problem(weights, opt_type="min")

        assert assignment.shape == (2, 2)
        assert assignment.sum(axis=1).all() == 1
        assert assignment.sum(axis=0).all() == 1
        assert objective == 5

    except Exception as e:
        pytest.fail(f"solve_assignment_problem square min test failed: {e}")


def test_solve_assignment_problem_rectangular():
    """It handles a rectangular assignment correctly."""
    try:
        import numpy as np
        from blackboxphaseopt.assignment_problem import solve_assignment_problem

        weights = np.array([[5, 1, 1],
                            [1, 5, 1]])

        assignment, objective = solve_assignment_problem(weights, opt_type="max")

        assert assignment.shape == (2, 3)
        assert assignment.sum(axis=1).max() <= 1
        assert assignment.sum(axis=0).max() <= 1

    except Exception as e:
        pytest.fail(f"solve_assignment_problem rectangular test failed: {e}")


# solve_assignment_problem_time_expanded

def test_solve_assignment_problem_time_expanded_basic():
    """It solves a simple time-expanded maximization case."""
    try:
        import numpy as np
        from blackboxphaseopt.assignment_problem import solve_assignment_problem_time_expanded

        weights = np.array([[[1, 2],
                             [3, 4]]])

        assignment, objective = solve_assignment_problem_time_expanded(weights)

        assert assignment.shape == (1, 2, 2)
        assert assignment.sum(axis=2).max() <= 1
        assert assignment.sum(axis=1).max() <= 1
        assert objective == 5

    except Exception as e:
        pytest.fail(f"solve_assignment_problem_time_expanded basic test failed: {e}")


def test_solve_assignment_problem_time_expanded_multiple_timesteps():
    """It solves multiple timesteps independently."""
    try:
        import numpy as np
        from blackboxphaseopt.assignment_problem import solve_assignment_problem_time_expanded

        weights = np.array([
            [[1, 2],
             [3, 4]],
            [[4, 3],
             [2, 1]]
        ])

        assignment, objective = solve_assignment_problem_time_expanded(weights)

        assert assignment.shape == (2, 2, 2)
        assert assignment.sum(axis=2).max() <= 1
        assert assignment.sum(axis=1).max() <= 1

    except Exception as e:
        pytest.fail(f"solve_assignment_problem_time_expanded multi-timestep test failed: {e}")


def test_solve_assignment_problem_time_expanded_penalty():
    """It accounts for assignment penalties correctly."""
    try:
        import numpy as np
        from blackboxphaseopt.assignment_problem import solve_assignment_problem_time_expanded

        weights = np.array([[[10, 1],
                             [1, 10]]])

        penalty = np.array([[[9, 0],
                             [0, 9]]])

        assignment, objective = solve_assignment_problem_time_expanded(
            weights,
            assignment_penalty_lambda=penalty
        )

        assert assignment.shape == (1, 2, 2)
        assert assignment.sum(axis=2).max() <= 1
        assert assignment.sum(axis=1).max() <= 1

    except Exception as e:
        pytest.fail(f"solve_assignment_problem_time_expanded penalty test failed: {e}")