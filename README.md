## Installation 
```
conda install malavp::blackboxphaseopt
```

## Building Conda package Locally
```
git clone https://github.com/Malav-P/bbpo
cd bbpo
conda build conda-recipe -c gurobi -c conda-forge
```

After the build completes, the package exists in you local channel. You can verify it by checking:
```
conda search blackboxphaseopt --use-local
```

If a package shows up you are good to go. Then you can install it into your virtual envs like so:

```
(venv)macbook@user123 % conda install blackboxphaseopt --use-local
```

## Developement

Create a conda env with the following packages
```yaml
    - python=3.13.3
    - numpy=2.3.5
    - conda-forge::heyoka.py=7.8.1
    - gurobi::gurobi=12.0.3
    - jax=0.7.2
    - pytest
```

Activate this environment. Install our package

```
git clone https://github.com/Malav-P/bbpo
cd bbpo
pip install -e .
```

Then try and import the package by running the following in terminal:

```
python -c "import blackboxphaseopt; print('import successful')"
```

Try running the basic tests from project root:
```
pytest -v
```