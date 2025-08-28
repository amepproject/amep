import os
import warnings
import logging
from typing import Collection, Iterable, Sequence, Optional, Union, Any, Dict, List, Tuple
LOGGERFORMAT = "%(levelname)s:%(name)s.%(funcName)s: %(message)s"
LOGGINGLEVEL = "INFO"


def check_path(path: str, extension: str) -> Tuple[str, str]:
    r"""
    Checks if the directory of a given path exists and separates between
    directory and file name.

    Parameters
    ----------
    path : str
        Path to check.
    extension : str
        File extension of the given path.

    Raises
    ------
    ValueError
        Raised if an invalid file extension has been identified.
    FileNotFoundError
        Raised if the directory does not exist.

    Returns
    -------
    directory : str
        Directory of the given path.
    filename : str
        File name in the given path.

    """
    # normalize path
    path = os.path.normpath(path)
    
    # get extension
    _, file_extension = os.path.splitext(path)
    
    # check extension
    if file_extension == extension:
        # split into directory and filename
        directory, filename = os.path.split(path)
        # set directory to current working directory if empty
        if directory == '':
            directory = os.getcwd()
    elif file_extension == '':
        directory = os.path.normpath(path)
        filename  = ''
    else:
        raise ValueError(
            f'''Incorrect file extension. Got {file_extension} instead
            of {extension}.'''
        )

    if os.path.exists(directory):
        return directory, filename
    else:
        raise FileNotFoundError(f'No such directory: {directory}')



def get_module_logger(mod_name):
    r"""
    Creates a module logger.

    Parameters
    ----------
    mod_name : str
        Module name. Always use `__name__`.

    Returns
    -------
    logger : logging.Logger
        Logger object.

    """
    # create logger
    logger = logging.getLogger(mod_name)

    # set logging level
    logger.setLevel(LOGGINGLEVEL)

    return logger


def get_class_logger(mod_name, class_name):
    r"""
    Creates a class logger.

    Parameters
    ----------
    mod_name : str
        Module name. Always use `__name__`.
    class_name : str
        Class name. Always use `self.__class__.__name__`.

    Returns
    -------
    logger : logging.Logger
        Logger object.

    """
    # create logger
    logger = logging.getLogger(mod_name + "." + class_name)

    # set logging level
    logger.setLevel(LOGGINGLEVEL)

    return logger