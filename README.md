```
cd ~/git/
git clone git@github.com:daeyun/dshinpy.git
# git clone https://github.com/daeyun/dshinpy.git
```

```
export PYTHONPATH="$HOME/git/dshinpy/:${PYTHONPATH}"
```

## Running tests

```
# Runs doctests and tests in ./tests.
py.test
```
