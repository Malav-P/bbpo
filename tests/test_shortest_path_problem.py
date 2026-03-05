import pytest


# solve_shortest_path

def test_solve_shortest_path_simple_chain():
    """It finds the correct shortest path in a simple 3-node chain."""
    try:
        import numpy as np
        from blackboxphaseopt.shortest_path_problem import solve_shortest_path

        W = np.array([
            [0, 1, 10],
            [0, 0, 1],
            [0, 0, 0]
        ])

        x_var, path, objective = solve_shortest_path(W, 0, 2)

        assert x_var.shape == (3, 3)
        assert objective == 2
        assert x_var[0, 1] == 1
        assert x_var[1, 2] == 1

    except Exception as e:
        pytest.fail(f"solve_shortest_path simple chain test failed: {e}")


def test_solve_shortest_path_direct_edge():
    """It prefers the direct edge when it is cheaper."""
    try:
        import numpy as np
        from blackboxphaseopt.shortest_path_problem import solve_shortest_path

        W = np.array([
            [0, 5, 1],
            [0, 0, 5],
            [0, 0, 0]
        ])

        x_var, path, objective = solve_shortest_path(W, 0, 2)

        assert objective == 1
        assert x_var[0, 2] == 1

    except Exception as e:
        pytest.fail(f"solve_shortest_path direct edge test failed: {e}")


def test_solve_shortest_path_has_source_and_target_flow():
    """It has outgoing flow from source and incoming flow to target."""
    try:
        import numpy as np
        from blackboxphaseopt.shortest_path_problem import solve_shortest_path

        W = np.array([
            [0, 1, 10],
            [0, 0, 1],
            [0, 0, 0]
        ])

        x_var, path, objective = solve_shortest_path(W, 0, 2)

        assert x_var[0].sum() >= 1
        assert x_var[:, 2].sum() >= 1

    except Exception as e:
        pytest.fail(f"solve_shortest_path flow presence test failed: {e}")


# solve_shortest_path_time_expanded

def test_solve_shortest_path_time_expanded_single_timestep():
    """It works correctly for a single timestep."""
    try:
        import numpy as np
        from blackboxphaseopt.shortest_path_problem import solve_shortest_path_time_expanded

        W = np.array([[
            [0, 1, 10],
            [0, 0, 1],
            [0, 0, 0]
        ]])

        x_var, path, objective = solve_shortest_path_time_expanded(W, 0, 2)

        assert x_var.shape == (1, 3, 3)
        assert objective == 2
        assert x_var[0, 0, 1] == 1
        assert x_var[0, 1, 2] == 1

    except Exception as e:
        pytest.fail(f"solve_shortest_path_time_expanded single timestep test failed: {e}")


def test_solve_shortest_path_time_expanded_multiple_timesteps():
    """It returns a feasible multi-timestep solution."""
    try:
        import numpy as np
        from blackboxphaseopt.shortest_path_problem import solve_shortest_path_time_expanded

        W = np.array([
            [
                [0, 5, 1],
                [0, 0, 5],
                [0, 0, 0]
            ],
            [
                [0, 1, 5],
                [0, 0, 1],
                [0, 0, 0]
            ]
        ])

        x_var, path, objective = solve_shortest_path_time_expanded(W, 0, 2)

        assert x_var.shape == (2, 3, 3)
        assert objective is not None
        assert objective > 0

        total_out_source = x_var[:, 0, :].sum()
        total_in_target = x_var[:, :, 2].sum()

        assert total_out_source >= 1
        assert total_in_target >= 1

    except Exception as e:
        pytest.fail(f"solve_shortest_path_time_expanded multi timestep test failed: {e}")


def test_solve_shortest_path_time_expanded_binary_solution():
    """It returns a binary solution matrix."""
    try:
        import numpy as np
        from blackboxphaseopt.shortest_path_problem import solve_shortest_path_time_expanded

        W = np.array([[
            [0, 1, 10],
            [0, 0, 1],
            [0, 0, 0]
        ]])

        x_var, path, objective = solve_shortest_path_time_expanded(W, 0, 2)

        assert np.all((x_var == 0) | (x_var == 1))

    except Exception as e:
        pytest.fail(f"solve_shortest_path_time_expanded binary test failed: {e}")