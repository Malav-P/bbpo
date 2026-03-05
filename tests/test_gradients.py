import sys
import types

import pytest


def import_select_gradients():
    """Import select_gradients while handling optional deps."""
    try:
        from blackboxphaseopt.gradients import select_gradients
        return select_gradients
    except Exception:
        import numpy as np

        sys.modules.pop("blackboxphaseopt.gradients", None)
        sys.modules.pop("blackboxphaseopt.dynamics", None)
        sys.modules.pop("jax", None)
        sys.modules.pop("jax.numpy", None)

        if "heyoka" not in sys.modules:
            sys.modules["heyoka"] = types.ModuleType("heyoka")

        jax = types.ModuleType("jax")
        jnp = types.ModuleType("jax.numpy")
        jnp.__dict__.update(np.__dict__)
        jax.numpy = jnp
        sys.modules["jax"] = jax
        sys.modules["jax.numpy"] = jnp

        from blackboxphaseopt.gradients import select_gradients
        return select_gradients


# select_gradients

def test_select_gradients_basic_one_hot():
    """It picks the right gradient for a one-hot assignment."""
    try:
        import numpy as np

        select_gradients = import_select_gradients()

        T = 3
        N = 4
        M = 5
        state_dim = 6

        gradients = np.arange(T * N * M * state_dim, dtype=float).reshape(T, N, M, state_dim)

        x = np.zeros((T, N, M), dtype=int)
        for t in range(T):
            for n in range(N):
                m = (t + n) % M
                x[t, n, m] = 1

        active = select_gradients(gradients, x)

        assert active.shape == (N, T, state_dim)

        for t in range(T):
            for n in range(N):
                m = (t + n) % M
                assert np.array_equal(active[n, t], gradients[t, n, m])

    except Exception as e:
        pytest.fail(f"select_gradients one-hot test failed: {e}")


def test_select_gradients_multiple_active_sums():
    """It sums gradients when multiple entries are active."""
    try:
        import numpy as np

        select_gradients = import_select_gradients()

        np.random.seed(0)
        T = 2
        N = 3
        M = 4
        state_dim = 6

        gradients = np.random.randn(T, N, M, state_dim)

        x = np.zeros((T, N, M), dtype=int)
        x[:, :, 0] = 1
        x[:, :, 2] = 1

        active = select_gradients(gradients, x)

        expected = gradients[:, :, 0, :] + gradients[:, :, 2, :]
        expected = expected.transpose(1, 0, 2)

        assert np.allclose(active, expected)

    except Exception as e:
        pytest.fail(f"select_gradients multi-active test failed: {e}")


def test_select_gradients_shape_mismatch_raises():
    """It raises an error if x has the wrong shape."""
    try:
        import numpy as np

        select_gradients = import_select_gradients()

        T = 2
        N = 2
        M = 3
        state_dim = 6

        gradients = np.zeros((T, N, M, state_dim))
        x_bad = np.ones((T, N, M - 1), dtype=int)

        with pytest.raises(ValueError):
            select_gradients(gradients, x_bad)

    except Exception as e:
        pytest.fail(f"select_gradients shape-mismatch test failed: {e}")


# compute_generalized_distances

def test_compute_generalized_distances_basic_value():
    """It computes the correct quadratic distance value."""
    try:
        import numpy as np
        from blackboxphaseopt.gradients import compute_generalized_distances

        N = 1
        M = 1
        T = 2

        states_x = np.array([[[1, 0, 0, 0, 0, 0],
                              [2, 0, 0, 0, 0, 0]]])

        states_y = np.array([[[0, 0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0, 0]]])

        Q = np.ones(6)

        dist, grad = compute_generalized_distances(states_x, states_y, Q)

        expected_t0 = (1 - 0) ** 2
        expected_t1 = (2 - 1) ** 2

        assert dist.shape == (T, N, M)
        assert np.isclose(dist[0, 0, 0], expected_t0)
        assert np.isclose(dist[1, 0, 0], expected_t1)

    except Exception as e:
        pytest.fail(f"compute_generalized_distances basic value test failed: {e}")


def test_compute_generalized_distances_gradient():
    """It computes the correct analytical gradient."""
    try:
        import numpy as np
        from blackboxphaseopt.gradients import compute_generalized_distances

        N = 1
        M = 1
        T = 1

        states_x = np.array([[[2, 0, 0, 0, 0, 0]]])
        states_y = np.array([[[1, 0, 0, 0, 0, 0]]])

        Q = np.ones(6)

        dist, grad = compute_generalized_distances(states_x, states_y, Q)

        expected_grad = 2 * (2 - 1)

        assert grad.shape == (T, N, M, 6)
        assert np.isclose(grad[0, 0, 0, 0], expected_grad)

    except Exception as e:
        pytest.fail(f"compute_generalized_distances gradient test failed: {e}")


def test_compute_generalized_distances_no_grad():
    """It returns None when gradient computation is disabled."""
    try:
        import numpy as np
        from blackboxphaseopt.gradients import compute_generalized_distances

        states_x = np.zeros((1, 1, 6))
        states_y = np.zeros((1, 1, 6))
        Q = np.ones(6)

        dist, grad = compute_generalized_distances(states_x, states_y, Q, compute_grad=False)

        assert grad is None

    except Exception as e:
        pytest.fail(f"compute_generalized_distances no-grad test failed: {e}")


# compute_projected_gradients

def test_compute_projected_gradients_none_reduction():
    """It returns the unreduced projected gradients when reduction is None."""
    try:
        import numpy as np
        from blackboxphaseopt.gradients import compute_projected_gradients

        N = 1
        T = 2

        gradients = np.ones((N, T, 6))
        states = np.ones((N, T, 6))

        projected = compute_projected_gradients(gradients, states, reduction=None)

        assert projected.shape == (N, T)

    except Exception as e:
        pytest.fail(f"compute_projected_gradients none reduction test failed: {e}")


def test_compute_projected_gradients_sum():
    """It sums projected gradients across time."""
    try:
        import numpy as np
        from blackboxphaseopt.gradients import compute_projected_gradients

        N = 1
        T = 3

        gradients = np.ones((N, T, 6))
        states = np.ones((N, T, 6))

        projected = compute_projected_gradients(gradients, states, reduction="sum")

        assert projected.shape == (N,)

    except Exception as e:
        pytest.fail(f"compute_projected_gradients sum test failed: {e}")


def test_compute_projected_gradients_invalid_reduction():
    """It raises a ValueError for invalid reduction input."""
    try:
        import numpy as np
        from blackboxphaseopt.gradients import compute_projected_gradients

        gradients = np.ones((1, 1, 6))
        states = np.ones((1, 1, 6))

        with pytest.raises(ValueError):
            compute_projected_gradients(gradients, states, reduction="invalid")

    except Exception as e:
        pytest.fail(f"compute_projected_gradients invalid reduction test failed: {e}")