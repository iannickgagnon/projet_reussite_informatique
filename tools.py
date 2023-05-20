
# External libraries
import numpy as np
from typing import List
from typing import Tuple
from typing import Iterable
from datetime import datetime
import matplotlib.pyplot as plt

# Internal constants
from constants import HOURS_PER_DAY


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


def bootstrap_generate_base_vector_from_bins(bins: Iterable) -> np.array:
    """
    Creates a base vector for bootstrap sampling based on histogram bins (i.e. counts). For example, the bins [2, 6, 4]
    becomes [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2].

    Args:
        bins (Iterable): The histogram bin counts.

    Returns:
        (np.array): The generated base vector as explained above.
    """

    return np.repeat(range(len(bins)), bins)


def bootstrap_generate_samples_from_bins(bins: Iterable,
                                         total_count: int,
                                         nb_bins: int,
                                         nb_samples: int = 1000) -> List[np.array]:
    """
    Creates bootstrap samples from bins. For example, the bins [2, 6, 4] resul in the base vector
    [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]. This vector is sampled randomly with replacement to generate said bootstrap
    samples. For example, we might obtain :

        bootstrap sample no.1: [1, 0, 0, 0, 2, 2, 1, 1, 1, 1, 2, 0]
        bootstrap sample no.2: [1, 0, 1, 2, 2, 2, 1, 1, 0, 1, 2, 1]
        ...
        bootstrap sample no.x: [2, 0, 1, 2, 1, 0, 0, 1, 2 ,1, 1, 0]

    In this example, the number of bins (nb_bins) is 3, so we count the number of zeros, the number of ones and the
    number of twos :

        bootstrap sample no.1: [4, 5, 3]
        bootstrap sample no.2: [2, 6, 4]
        ...
        bootstrap sample no.x: [4, 5, 3]

    Since each column represents a bin and each bin represents a survey element distribution (e.g. a question of an
    answer), the samples are split column-wise and returned so that confidence intervals can be calculated for each :

        [[4, 2, ..., 4], [5, 6, ..., 5], [3, 4, ..., 3]]

    Args:
        bins (Iterable): Bin counts.
        total_count (int): The sum of bin counts.
        nb_bins (int): The number of bins.
        nb_samples (int, optional): The number of bootstrap samples to generate. Defaults to 1e3.

    Returns:
        (List[np.array]): The generated bootstrap samples for each bin.
    """

    # Generate base vector for bootsrap
    base = bootstrap_generate_base_vector_from_bins(bins)

    # Initialize counts
    bootstrap_counts = np.zeros((nb_samples, nb_bins))

    for i in range(nb_samples):

        # Pick a random sample with replacement
        sample = np.random.choice(base, size=total_count, replace=True)

        # Add counts for each bin of the current sample
        for j in range(nb_bins):
            bootstrap_counts[i, j] = np.sum(sample == j)

    # Extract columns
    columns = [column.squeeze() for column in np.split(bootstrap_counts, nb_bins, axis=1)]

    # Return columns
    return columns


def confidence_interval_to_string(value: (int, float),
                                  lower_bound: (int, float),
                                  upper_bound: (int, float),
                                  width: int = 100):
    """
    Converts a confidence interval to a string representation.

    Args:
        value (int or float): The value within the confidence interval.
        lower_bound (int or float): The lower bound of the confidence interval.
        upper_bound (int or float): The upper bound of the confidence interval.
        width (int, optional): The length of the visual bar. Defaults to 100.

    Returns:
        (str): The string representation of the confidence interval.
    """

    if lower_bound == upper_bound:
        return ''

    # Determine position of the value and the bounds
    position_val = value // 2
    position_lower_bound = lower_bound // 2
    position_upper_bound = upper_bound // 2

    # Store positions and corresponding symbols
    positions = (position_val, position_lower_bound, position_upper_bound)
    symbols = ('*', '|', '|')

    # Initialize string representation
    confidence_interval_string = '-' * width

    # Add symbols
    for position, symbol in zip(positions, symbols):
        confidence_interval_string = confidence_interval_string[:position] + symbol + confidence_interval_string[position + 1:]

    # Add end caps
    return '[' + confidence_interval_string + ']'


def calculate_dates_difference_hours(start_date_str, end_date_str):

    # Parse start and end dates
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Calculate time difference
    time_difference = end_date - start_date

    # Return difference in number of hours
    return time_difference.days * HOURS_PER_DAY