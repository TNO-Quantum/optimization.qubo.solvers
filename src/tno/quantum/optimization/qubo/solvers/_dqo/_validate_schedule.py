from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def check_ndim(schedule: NDArray[Any], ndim: int) -> None:
    """Check the number of dimension of the provided `schedule`.

    Args:
      schedule: NDArray to check the number of dimension of.
      ndim: expected number of dimensions.

    Raises:
        ValueError: if the number of dimensions of `schedule` differs from `ndim`.
    """
    if schedule.ndim != ndim:
        msg = (
            f"expected {ndim}-dimensional array, "
            f"but {schedule.ndim}-dimensional array was given"
        )
        raise ValueError(msg)


def check_shape(schedule: NDArray[Any], shape: tuple[int, ...]) -> None:
    """Check the `shape` of a 2D `schedule`.

    Args:
      schedule: 2D Array to check the shape of.
      shape: expected shape of the schedule.

    Raises:
        ValueError: if the shape of `schedule` does not match `shape`.
    """
    if schedule.shape != shape:
        msg = f"A 2D schedule must have shape {shape}, got {schedule.shape}"
        raise ValueError(msg)


def check_length(schedule: NDArray[Any], expected_length: int) -> None:
    """Check the length of a 1D `schedule`.

    Args:
      schedule: 1D Array to check the length of.
      expected_length: expected length of the schedule.

    Raises:
        ValueError: if the length of `schedule` does not match `expected_length`.
    """
    if len(schedule) != expected_length:
        msg = f"Schedule must have length {expected_length}, got {len(schedule)}"
        raise ValueError(msg)


def check_initial_ham_schedule(schedule: NDArray[np.number]) -> None:
    """Check if the initial point in the provided `schedule` is larger than the last.

    Args:
      schedule: 1D Array to check.

    Raises:
      ValueError: when the initial point of the schedule is smaller or equal to the last
        point in the schedule.
    """
    if schedule[0] <= schedule[-1]:
        msg = (
            "The first row of a 2D schedule represents the schedule for the initial "
            "Hamiltonian, whose first value should be much larger than the last value "
            "in the schedule."
        )
        raise ValueError(msg)


def check_problem_ham_schedule(schedule: NDArray[np.number]) -> None:
    """Check if the initial point in the provided `schedule` is smaller than the last.

    Args:
      schedule: 1D Array to check.

    Raises:
      ValueError: when the initial point of the schedule is larger or equal to the last
        point in the schedule.
    """
    if schedule[0] >= schedule[-1]:
        msg = (
            "The second row of a 2D schedule represents the schedule for the problem"
            "Hamiltonian, whose first value should be much smaller than the last value "
            "in the schedule."
        )
        raise ValueError(msg)


def check_schedule_monotonicity(schedule: NDArray[np.number]) -> None:
    """Check the monotonicity conditions of a 2D Hamiltonian schedule.

    This function checks that the initial Hamiltonian's schedule starts with a
    larger value than the problem Hamiltonian's schedule and ends with a smaller value.
    If these conditions are not met, a ``ValueError`` is raised.

    Args:
      schedule: 2D-Array, where the first row corresponds to the initial Hamiltonian
        schedule and the second row to the problem Hamiltonian schedule.
    """
    if schedule[0, 0] < schedule[1, 0]:
        msg = (
            "In a 2D schedule, the first value of the schedule of the initial "
            "Hamiltonian should be much larger than the first value of the schedule of "
            "the problem Hamiltonian."
        )
        raise ValueError(msg)

    if schedule[0, -1] > schedule[1, -1]:
        msg = (
            "In a 2D schedule, the last value of the schedule of the initial "
            "Hamiltonian should be much smaller than the last value of the schedule of "
            "the problem Hamiltonian."
        )
        raise ValueError(msg)


def check_1d_schedule_bounds(schedule: NDArray[np.number]) -> None:
    """Check if all values in the `schedule` are within $[0, 1]$.

    Args:
      schedule: 1D Array to check.

    Raises:
      ValueError: when any value in the `schedule` is not in $[0, 1]$.
    """
    if np.min(schedule) < 0.0 or np.max(schedule) > 1.0:
        msg = "All values in a 1D schedule must be in the range [0, 1]"
        raise ValueError(msg)


def check_1d_schedule_start_and_end(schedule: NDArray[np.number]) -> None:
    """Check if the initial point in the provided `schedule` is smaller than the last.

    Args:
      schedule: 1D Array to check.

    Raises:
      ValueError: when the initial point of the schedule is larger or equal to the last
        point in the schedule.
    """
    if schedule[0] >= schedule[-1]:
        msg = "The first value in a 1D schedule should be smaller than the last value"
        raise ValueError(msg)
