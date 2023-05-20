
# External libraries
import numpy as np
from typing import Tuple
from typing import Iterable
import matplotlib.pyplot as plt


def bootstrap_calculate_confidence_interval(base_vector: np.array, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculates (1 - alpha) confidence intervals (CI) for a given dataset.

    Args:
      base_vector (np.array): Data to calculate the CI for.
      alpha (float): Significance level of the desired interval.

    Returns:
      (Tuple[float, float]): Lower and upper bounds of the CI.
    """

    # Sort in ascending order
    sorted_data = np.sort(base_vector)

    # (1 - alpha) confidence bounds
    lower_bound = alpha / 2
    upper_bound = 1 - lower_bound

    # Confidence bounds indexes
    lower_index = int(lower_bound * base_vector.shape[0])
    upper_index = int(upper_bound * base_vector.shape[0])

    return sorted_data[lower_index], sorted_data[upper_index]


def letter_grade_to_points(grade: str) -> float:
    """
    Convert a letter grade to a point value.

    Args:
        grade (str): A string representing the letter grade.

    Returns:
        (float): The equivalent point value.

    Raises:
        ValueError: If the input grade is not a valid letter grade.
    """

    if grade == 'A+':
        return 4.3
    elif grade == 'A':
        return 4.0
    elif grade == 'A-':
        return 3.7
    elif grade == 'B+':
        return 3.3
    elif grade == 'B':
        return 3.0
    elif grade == 'B-':
        return 2.7
    elif grade == 'C+':
        return 2.3
    elif grade == 'C':
        return 2.0
    elif grade == 'C-':
        return 1.7
    elif grade == 'D+':
        return 1.3
    elif grade == 'D':
        return 1.0
    elif grade == 'E':
        return 0.0
    else:
        raise ValueError(f'Invalid grade ({grade})')


def plot_confidence_intervals(low: list, mid: list, high: list, x_labels: list =None):
    """
    Plots confidence intervals with custom x labels.

    Args:
      low (list): List of lower bounds.
      mid (list): List of middle points.
      high (list): List of upper bounds.
      xticks (list, optional): List of user-defined x labels. Defaults to None.

    Returns:
      Nothing.
    """

    TIP_LINE_WIDTH = 0.25       # Width of the top and bottom lines
    LINE_COLOR = 'steelblue'    # Line color
    DOT_COLOR = 'indianred'     # Color of center dot

    # Initialize figure and axes
    fig, ax = plt.subplots()

    for i in range(len(low)):

        # Vertical line
        ax.plot([i, i], [low[i], high[i]], color=LINE_COLOR)

        # Tip lines
        left = i - 0.5 * TIP_LINE_WIDTH
        right = i + 0.5 * TIP_LINE_WIDTH

        # Tip lines
        ax.plot([left, right], [high[i], high[i]], color=LINE_COLOR)
        ax.plot([left, right], [low[i], low[i]], color=LINE_COLOR)

        # Center dot
        ax.plot(i, mid[i], 'o', color=DOT_COLOR)

    # Change x labels
    if x_labels is not None:
        plt.xticks(list(range(len(low))), x_labels)

    return ax


def any_token_in_string(string: str, tokens: Iterable[str]) -> bool:
    """
    Check if any of the given tokens is present in the given string.

    Args:
        string (str): The string to check.
        tokens (Iterable[str]): The tokens to look for.

    Returns:
        (bool): True if any of the tokens is present in the string, False otherwise.
    """

    return any(token in string for token in tokens)
