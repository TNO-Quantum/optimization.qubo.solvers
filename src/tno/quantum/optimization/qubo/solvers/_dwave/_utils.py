"""This module contains utility functions for the D-Wave solvers."""

import logging
import time
import warnings
from http.client import RemoteDisconnected
from typing import Any, Callable, TypeVar

from urllib3.exceptions import ProtocolError

from tno.quantum.optimization.qubo.components import ResultInterface, Solver

T = TypeVar("T")

_singletons = {}


def get_singleton(cls: type[T], *args: Any, **kwargs: Any) -> T:
    """Construct singleton instance of class.

    Args:
        cls: Class to construct instance of.
        args: Positional arguments for constructor of class.
        kwargs: Keyword arguments for constructor of class.

    Returns:
        Instance of class.
    """
    key = f"{cls.__module__}.{cls.__name__}({args}, {kwargs})"
    if key not in _singletons:
        _singletons[key] = cls(*args, **kwargs)
    return _singletons[key]


RESULT_TYPE = TypeVar("RESULT_TYPE", bound=ResultInterface)


def retry_on_network_errors(
    func: Callable[..., RESULT_TYPE],
) -> Callable[..., RESULT_TYPE]:
    """Decorator that retries to run function on network errors.

    This decorator runs the given function `func` a number of times before raising a
    network error: ``RemoteDisconnected``, ``ProtocolError`` or ``ConnectionError``.
    Waits 15 seconds in between attempts. The number of attempts is taken as
    `self.num_attempts`. This method also sets the field `num_attempts` of the result.

    Args:
        func: Method to run. Typically the `_solve` method of a ``Solver`` class.

    Returns:
        Wrapper of `func`.
    """

    def wrapper(self: Solver[Any], *args: Any, **kwargs: Any) -> RESULT_TYPE:
        num_attempts = 1
        if hasattr(self, "num_attempts"):
            num_attempts = self.num_attempts
        else:
            msg = (
                "Decorator `retry_on_network_errors` used, "
                "but solver has no attribute `num_attempts`"
            )
            warnings.warn(msg, stacklevel=2)

        for attempt in range(1, num_attempts + 1):
            try:
                result = func(self, *args, **kwargs)
                break
            except (
                RemoteDisconnected,
                ProtocolError,
                ConnectionError,
            ) as exception:
                msg = f"Failed to establish connection (attempt {attempt})"
                if attempt < num_attempts:
                    msg += ". Retrying in 15 seconds .."
                logging.getLogger(self.__class__.__name__).warning(msg)
                caught_exception = exception
                time.sleep(15)  # try again in 15 seconds
        else:
            # When all attempts failed, re-raise the last exception
            raise RemoteDisconnected(str(caught_exception)) from caught_exception

        # Set number of attempts property on result
        result.num_attempts = attempt

        return result

    return wrapper
