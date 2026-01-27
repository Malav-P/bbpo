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