import pytest

def test_import():
    """Test that package imports successfully"""
    try:
        import blackboxphaseopt
        assert blackboxphaseopt is not None
    except ImportError as e:
        pytest.fail(f"Failed to import blackboxphaseopt: {e}")
