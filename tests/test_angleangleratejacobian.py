import sys
import types
import pytest


# angleanglerate_jacobian

def import_angleanglerate_jacobian():
    """Import angleanglerate_jacobian while handling JAX dependency."""
    try:
        from blackboxphaseopt.utilities import angleanglerate_jacobian
        return angleanglerate_jacobian
    except Exception:
        import numpy as np

        sys.modules.pop("jax", None)
        sys.modules.pop("jax.numpy", None)

        jax = types.ModuleType("jax")
        jnp = types.ModuleType("jax.numpy")
        jnp.__dict__.update(np.__dict__)

        jax.numpy = jnp

        sys.modules["jax"] = jax
        sys.modules["jax.numpy"] = jnp

        from blackboxphaseopt.utilities import angleanglerate_jacobian
        return angleanglerate_jacobian


def test_angleanglerate_jacobian_shape_and_blocks():
    """It returns a 6x6 matrix with correct block structure."""
    try:
        import numpy as np

        angleanglerate_jacobian = import_angleanglerate_jacobian()

        x = np.zeros(6)
        y = np.array([1., 1., 1., 0., 0., 0.])

        H = np.array(angleanglerate_jacobian(x, y))

        assert H.shape == (6, 6)

        top_right = H[0:3, 3:6]
        bottom_right = H[3:6, 3:6]
        top_left = H[0:3, 0:3]

        assert np.allclose(top_right, np.zeros((3, 3)))
        assert np.allclose(bottom_right, top_left)

    except Exception as e:
        pytest.fail(f"angleanglerate_jacobian block structure test failed: {e}")


def test_angleanglerate_jacobian_radial_case():
    """It produces correct structure when relative position is along x-axis."""
    try:
        import numpy as np

        angleanglerate_jacobian = import_angleanglerate_jacobian()

        x = np.zeros(6)
        y = np.array([2., 0., 0., 0., 0., 0.])

        H = np.array(angleanglerate_jacobian(x, y))

        H11 = H[0:3, 0:3]

        assert np.isclose(H11[0, 0], 0.0)
        assert np.isclose(H11[1, 1], 1/2)
        assert np.isclose(H11[2, 2], 1/2)

    except Exception as e:
        pytest.fail(f"angleanglerate_jacobian radial case test failed: {e}")


def test_angleanglerate_jacobian_finite_difference():
    """It responds to small perturbations in position."""
    try:
        import numpy as np

        angleanglerate_jacobian = import_angleanglerate_jacobian()

        x = np.zeros(6)
        y = np.array([1., 2., 3., 0., 0., 0.])

        H = np.array(angleanglerate_jacobian(x, y))

        eps = 1e-6
        dx = np.array([eps, 0., 0., 0., 0., 0.])

        H_perturbed = np.array(angleanglerate_jacobian(x + dx, y))

        approx_derivative = (H_perturbed - H) / eps

        assert np.linalg.norm(approx_derivative) > 0

    except Exception as e:
        pytest.fail(f"angleanglerate_jacobian finite difference test failed: {e}")