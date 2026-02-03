from logging import config
import pytest

def test_import():
    """Test that package imports successfully"""
    try:
        import blackboxphaseopt
        assert blackboxphaseopt is not None
    except ImportError as e:
        pytest.fail(f"Failed to import blackboxphaseopt: {e}")


def test_heyoka():
    """Test that heyoka is available and works appropriately"""

    try:
        import numpy as np
        from blackboxphaseopt import dynamics
        from blackboxphaseopt.constants import CR3BP_MU, Config

        initial_state_y = np.array([ 
                    0.8027692908754149,
                    0.0,
                    0.0,
                    -1.1309830924549648e-14,
                    0.33765564334938736,
                    0.0
                ])

        config = Config(
            period = 3.225,
            n_points = 215,
            initial_state_target = np.tile(initial_state_y, (2, 1)),
            initial_state_target_phases = (0, 0.5)
        )

        ta = dynamics.build_taylor_cr3bp(mu=CR3BP_MU)

        _, states_y = dynamics.gen_state_history(ta=ta,
                                initial_state=config.initial_state_target,
                                time=config.period,
                                n_points=config.n_points,
                                phase=config.initial_state_target_phases)


        assert states_y.shape == (2, config.n_points, 6)

    # except on any error
    except Exception as e:
        pytest.fail(f"Heyoka test failed: {e}")
    
