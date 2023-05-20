
# External libraries
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

# Internal constants
from constants import PATH_MAIN_PLOT_STYLE


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

    for i in range(len(low)):
        # Vertical line
        plt.plot([i, i], [low[i], high[i]], color=LINE_COLOR)

        # Tip lines
        left = i - 0.5 * TIP_LINE_WIDTH
        right = i + 0.5 * TIP_LINE_WIDTH

        # Tip lines
        plt.plot([left, right], [high[i], high[i]], color=LINE_COLOR)
        plt.plot([left, right], [low[i], low[i]], color=LINE_COLOR)

        # Center dot
        plt.plot(i, mid[i], 'o', color=DOT_COLOR)

    # Change x labels
    if x_labels is not None:
        plt.xticks(list(range(len(low))), x_labels)


if __name__ == '__main__':

    # TODO: Clean up

    # CIs for Abandon

    '''low = [42, 64, 67, 77, 80, 83]
    mid = [53, 79, 80, 91, 92, 94]
    high = [68, 93, 94, 100, 100, 100]
    
    with plt.style.context(PATH_MAIN_PLOT_STYLE):

        plot_confidence_intervals(low, mid, high, x_labels=['1A', '1B', '1C', '2', '3', '4'])
        plt.title('Intervalles de confiance 95% pour l\'abandon')
        plt.xlabel('Modèle')
        plt.ylabel('Précision [%]')
        plt.ylim([40, 102.5])
        plt.gca().yaxis.grid(True, linestyle='--', alpha=0.625)
        plt.show()'''


    # CIs for Fail

    low = [43, 54, 52, 70, 71, 78]
    mid = [52, 67, 68, 84, 86, 90]
    high = [62, 81, 85, 86, 100, 100]
    
    with plt.style.context(PATH_MAIN_PLOT_STYLE):

        plot_confidence_intervals(low, mid, high, x_labels=['1A', '1B', '1C', '2', '3', '4'])
        plt.title('Intervalles de confiance 95% pour l\'échec')
        plt.xlabel('Modèle')
        plt.ylabel('Précision [%]')
        plt.ylim([40, 102.5])
        plt.gca().yaxis.grid(True, linestyle='--', alpha=0.625)
        plt.show()


    # CIs for Fail
    '''
    low = [95, 95, 90, 95, 95, 96]
    mid = [98, 95, 93, 97, 97, 99]
    high = [100, 96, 95, 100, 100, 100]

    with plt.style.context(PATH_MAIN_PLOT_STYLE):
        plot_confidence_intervals(low, mid, high, x_labels=['1A', '1B', '1C', '2', '3', '4'])
        plt.title('Intervalles de confiance 95% pour le succès')
        plt.xlabel('Modèle')
        plt.ylabel('Précision [%]')
        plt.ylim([40, 102.5])
        plt.show()
    '''


