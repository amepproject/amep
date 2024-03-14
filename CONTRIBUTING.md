# Contributing to AMEP

Contributions are always welcome and the **AMEP** development team appreciates 
any help you give. If you have any comments, ideas, or if you want to 
contribute to this project, please contact the **AMEP** developers via e-mail 
(support@amepproject.de) or by starting a discussion on github. When 
contributing to **AMEP**, please follow the coding guidelines specified below.

## Pull requests

Contributions are welcome via pull requests. Multiple developers and/or users 
will review requested changes and make comments. The lead developer(s) will 
merge your contribution into the main branch after the review is completed and 
approved.

When submitting a pull request, we ask you to check the following:

* Unit tests, documentation, and code style are in order and follow the 
guidelines specified below
* It's also OK to submit work in progress if you're unsure of what this exactly 
means, in which case you'll likely be asked to make some further changes.
* The contributed code will be licensed under AMEP's license. If you did not write the 
code yourself, you ensure the existing license is compatible and include the 
license information in the contributed files, or obtain permission from the 
original author to relicense the contributed code.

## Coding guidelines

Python code in **AMEP** should follow the guidelines specified in 
[PEP8](https://peps.python.org/pep-0008/) and should provide 
[type hints](https://peps.python.org/pep-0484/). For the docstrings, **AMEP** 
uses the [NumPy guidelines](https://numpydoc.readthedocs.io/en/latest/format.html). 
Here is a minimal example on how a function should be defined following our 
guidelines:

```python
import numpy as np

def myfunc(
        x: float | np.ndarray,
        a: float = 1.0) -> float | np.ndarray:
    r"""Precise description of the method.
    
    Notes
    -----
    The function is defined by
    
    .. math::
    
        f(x) = 2x + a
    
    as discussed in Ref. [1]_.
    
    References
    ----------
    .. [1] Author, "title", journal, volume, page, year.
    
    Parameters
    ----------
    x : float or np.ndarray
        Description of parameter x.
    a : float, optional
        Description of parameter a.
        The default is 1.0.
    
    Returns
    -------
    float or np.ndarray
        Description of return value.
    
    Examples
    --------
    >>> x = 2.0
    >>> f(x, a = 4)
    8.0
    >>>
    
    """
    return 2*x + a
```

Additionally, we use the following naming conventions for methods and classes:

```python
# global variable
DTYPE=float

# local variable
dtype=float

# method with short name
def shortname():
    pass

# method with long name
def very_long_method_name():
    pass

# class with short name
class Shortname:
    def __init__(self):
        pass

# class with long name
class VeryLongClassName:
    def __init__(self):
        pass
```

The names of variables, methods, and classes should be self-explaining, as 
specific as possible, and as short as possible.

## Unit tests

For each module, there should be a set of unit tests, which test the correct 
behavior of the included methods and classes. These unit tests should be 
summarized in one Python file per module and added to the `test` directory of 
the **AMEP** repository. The tests should be short and simple, i.e., each test 
method should only test a single function. These tests should complete as fast 
as possible. Whenever a new method is added to **AMEP**, it is required to also 
add the corresponding test method.

## Profiling code

We always recommend to profile your code. This is very important for writing 
efficient and fast code. Here is some information about how to profile your 
code in a Jupyter notebook: First, you need to install the following libraries 
(see also https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html):

`conda install -c conda-forge line_profiler`

`conda install -c conda-forge memory_profiler`

In the Jupyter notebook, you have to load the profilers:

```
%load_ext memory_profiler
%load_ext line_profiler
```

Assuming you want to profile a method `myfunc` stored in a file `mymodule.py`. 
Then, you can get the total run time and the memory consumption by

```
import mymodule
%timeit mymodule.myfunc()
%memit mymodule.myfunc()
```

You can also get the time and memory consumption line by line with the 
following code, respectively:

`%lprun -f mymodule.myfunc mymodule.myfunc()`

`%mprun -f mymodule.myfunc mymodule.myfunc()`