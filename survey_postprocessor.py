
# External libraries
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from typing import (
  Iterable,
  Tuple,
  List,
)

# Internal libraries
from course import Course

# Internal constants
from constants import (
    COURSE_OUTCOMES,
    COURSE_NB_OUTCOMES,
    SURVEY_NUMERICAL_QUESTIONS_INDEX,
)


def bootstrap_generate_base_vector_from_bins(bins: Iterable) -> np.array:
    """
    Creates a base vector for bootstrap sampling based on histogram bins. For example, the bins [2, 6, 4] becomes
    [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2].

    Args:
        bins (Iterable): The histogram bin counts.

    Returns:
        (np.array): The generated base vector as explained above.
    """

    # Calculate size of base vector
    n = sum(bins)

    # Initialize base vector
    base_vector = np.zeros(n)

    # Build base vector
    m = 0
    for i, bin_length in enumerate(bins):
        for _ in range(bin_length):
          base_vector[m] = i
          m += 1

    return base_vector


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
    position_val = int(width * value / (width * 2))
    position_lower_bound = int(width * lower_bound / (width * 2))
    position_upper_bound = int(width * upper_bound / (width * 2))

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


if __name__ == '__main__':

    # TODO: Remove
    with open('courses.pkl', 'rb') as file:
        courses = pickle.load(file)

    # Build compiled results data structure
    questions_and_outcomes = Course.build_compiled_survey_results_data_structure()

    # Initialize list of worked hours and outcome pairs
    nb_hours_worked_outcome_pairs = []

    # Analyser tous les sigles confondus
    is_found = 0
    for course in courses:
        for answers in course.surveys:
            for student in course.students:
                if student.name == answers.student_name:
                    for question_index, answer in enumerate(answers):
                        if question_index in SURVEY_NUMERICAL_QUESTIONS_INDEX:
                            nb_hours_worked_outcome_pairs.append((answers[question_index], student.get_outcome()))
                        else:
                            questions_and_outcomes[question_index][answer][student.get_outcome()] += 1

    # Count the total number of answers
    total_counts = sum((sum(d.values()) for d in questions_and_outcomes[0].values()))

    # Initialize data structure for box plots
    box_plot_data = []

    # Parse answers
    for question_index, question in enumerate(questions_and_outcomes.values()):

        # No confidence intervals for numerical questions
        if question_index in SURVEY_NUMERICAL_QUESTIONS_INDEX:
            continue

        # Show question number
        print(f'Question no.{question_index + 1}\n')

        # Generate bins from question dictionary
        bins_question = [sum(d.values()) for d in question.values()]

        # Calculate the number of bins
        nb_bins = len(bins_question)

        # Generate bootstrap samples
        bootstrap_samples_answers = bootstrap_generate_samples_from_bins(bins_question, total_counts, nb_bins)

        for answer_key, value, sample in zip(question.keys(), bins_question, bootstrap_samples_answers):

            # Calculate confidence interval for current question
            lower, upper = bootstrap_calculate_confidence_interval(sample)

            # Transform in percentages
            value = value / total_counts * 100
            lower = lower / total_counts * 100
            upper = upper / total_counts * 100

            # Show proportions for each answer
            ci_str = confidence_interval_to_string(value, lower, upper)
            print(f'\tANSWER \'{answer_key}\': {value:<4.1f}% [{lower:4.1f}, {upper:4.1f}] {ci_str}\n')

            # Parse outcomes
            outcomes = questions_and_outcomes[question_index][answer_key]

            # Generate bins from question dictionary
            bins_outcomes = list(outcomes.values())
            total_outcomes = sum(bins_outcomes)

            # Generate bootstrap samples
            bootstrap_outcomes = bootstrap_generate_samples_from_bins(bins_outcomes,
                                                                      sum(bins_outcomes),
                                                                      COURSE_NB_OUTCOMES)

            outcome_index = 0
            for outcome_key, value_outcome, sample_outcome in zip(COURSE_OUTCOMES, bins_outcomes, bootstrap_outcomes):

                # Add to box plot data
                box_plot_data.append((question_index,
                                      answer_key,
                                      outcome_key,
                                      sample_outcome / total_outcomes * 100,
                                      bins_outcomes[outcome_index]))

                outcome_index += 1

                # Calculate confidence interval for current question
                lower, upper = bootstrap_calculate_confidence_interval(sample_outcome)

                # Transform in percentages
                value_outcome = value_outcome / total_outcomes * 100
                lower = lower / total_outcomes * 100
                upper = upper / total_outcomes * 100

                # Show proportions for each answer
                ci_str = confidence_interval_to_string(value_outcome, lower, upper)
                print(f'\t\t\'{outcome_key:7s}\': {value_outcome:<4.1f}% [{lower:4.1f}, {upper:4.1f}]\t{ci_str}')

            print()

    # SITUATION FINANCIERE ECHEC
    with plt.style.context('./images/main_plot_style.mplstyle'):

        fig, ax = plt.subplots()

        box_plot_data_indexes = (71, 74, 77)

        # Sample sizes
        n1 = box_plot_data[box_plot_data_indexes[0]][4]
        n2 = box_plot_data[box_plot_data_indexes[1]][4]
        n3 = box_plot_data[box_plot_data_indexes[2]][4]
        n_total = n1 + n2 + n3

        ax.boxplot([box_plot_data[i][3] for i in box_plot_data_indexes])

        ax.yaxis.grid(True, linestyle='--', alpha=0.625)

        ax.set_xticklabels((f'Aisée (n = {n1})', f'Satisfaisante (n = {n2})', f'Précaire (n = {n3})'))
        plt.title(f'Relation entre la situation financière et l\'échec (n = {n_total})')
        plt.ylabel('Taux d\'échec [%]')

        plt.show()

    # ENFANTS VS ABANDON
    with plt.style.context('./images/main_plot_style.mplstyle'):

        fig, ax = plt.subplots()

        box_plot_data_indexes = (57, 60)

        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(box_plot_data[box_plot_data_indexes[0]][3],
                                               box_plot_data[box_plot_data_indexes[1]][3])
        test_str = f'Test Mann-Whitney U : p < 0.001'

        # Sample sizes
        n1 = box_plot_data[box_plot_data_indexes[0]][4]
        n2 = box_plot_data[box_plot_data_indexes[1]][4]
        n_total = n1 + n2

        ax.boxplot([box_plot_data[i][3] for i in box_plot_data_indexes])

        ax.yaxis.grid(True, linestyle='--', alpha=0.625)

        ax.set_xticklabels((f'Enfants (n = {n1})', f'Pas d\'enfants (n = {n2})'))
        plt.title(f'Relation entre le fait d\'avoir des enfants et l\'abandon (n = {n_total})')
        plt.ylabel('Taux d\'abandon [%]')

        plt.show()

    # LANGUE MATERNELLE ET ABANDON
    with plt.style.context('./images/main_plot_style.mplstyle'):
        fig, ax = plt.subplots()

        box_plot_data_indexes = (48, 54)

        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(box_plot_data[box_plot_data_indexes[0]][3],
                                               box_plot_data[box_plot_data_indexes[1]][3])
        test_str = f'Test Mann-Whitney U : p < 0.001'

        # Sample sizes
        n1 = box_plot_data[box_plot_data_indexes[0]][4]
        n2 = box_plot_data[box_plot_data_indexes[1]][4]
        n_total = n1 + n2

        ax.boxplot([box_plot_data[i][3] for i in box_plot_data_indexes])

        ax.yaxis.grid(True, linestyle='--', alpha=0.625)

        ax.set_xticklabels((f'Français (n = {n1})', f'Autre (n = {n2})'))
        plt.title(f'Relation entre la langue parlée à la maison et l\'abandon (n = {n_total})')
        plt.ylabel('Taux d\'abandon [%]')

        plt.show()

    # LANGUE MATERNELLE ET ABANDON
    with plt.style.context('./images/main_plot_style.mplstyle'):

        fig, ax = plt.subplots()

        box_plot_data_indexes = (39, 45)

        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(box_plot_data[box_plot_data_indexes[0]][3], box_plot_data[box_plot_data_indexes[1]][3])
        test_str = f'Test Mann-Whitney U : p < 0.001'

        # Sample sizes
        n1 = box_plot_data[box_plot_data_indexes[0]][4]
        n2 = box_plot_data[box_plot_data_indexes[1]][4]
        n_total = n1 + n2

        ax.boxplot([box_plot_data[i][3] for i in box_plot_data_indexes])

        ax.yaxis.grid(True, linestyle='--', alpha=0.625)

        ax.set_xticklabels((f'Français (n = {n1})', f'Autre (n = {n2})'))
        plt.title(f'Relation entre la langue maternelle et l\'abandon (n = {n_total})')
        plt.ylabel('Taux d\'abandon [%]')

        plt.show()

    # ETUDES VS ETUDES ET TRAVAIL
    with plt.style.context('./images/main_plot_style.mplstyle'):

        fig, ax = plt.subplots()

        box_plot_data_indexes = (32, 38)

        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(box_plot_data[box_plot_data_indexes[0]][3], box_plot_data[box_plot_data_indexes[1]][3])
        test_str = f'Test Mann-Whitney U : p < 0.001'

        # Sample sizes
        n1 = box_plot_data[box_plot_data_indexes[0]][4]
        n2 = box_plot_data[box_plot_data_indexes[1]][4]
        n_total = n1 + n2

        ax.boxplot([box_plot_data[i][3] for i in box_plot_data_indexes])

        ax.yaxis.grid(True, linestyle='--', alpha=0.625)

        ax.set_xticklabels((f'Études (n = {n1})', f'Études + travail (n = {n2})'))
        plt.title(f'Relation entre le travail et l\'échec (n = {n_total})')
        plt.ylabel('Taux d\'échec [%]')

        plt.show()

    # REPRISE VS ECHEC
    with plt.style.context('./images/main_plot_style.mplstyle'):

        fig, ax = plt.subplots()

        box_plot_data_indexes = (2, 5)

        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(box_plot_data[box_plot_data_indexes[0]][3], box_plot_data[box_plot_data_indexes[1]][3])
        test_str = f'Test Mann-Whitney U : p < 0.001'

        # Sample sizes
        n1 = box_plot_data[box_plot_data_indexes[0]][4]
        n2 = box_plot_data[box_plot_data_indexes[1]][4]
        n_total = n1 + n2

        ax.boxplot([box_plot_data[i][3] for i in box_plot_data_indexes])

        ax.yaxis.grid(True, linestyle='--', alpha=0.625)

        ax.set_xticklabels((f'Reprise (n = {n1})', f'Pas reprise (n = {n2})'))
        plt.title(f'Relation entre la reprise d\'un cours et l\'échec (n = {n_total})')
        plt.ylabel('Taux d\'échec [%]')

        plt.show()

    # REPRISE VS ABANDON
    with plt.style.context('./images/main_plot_style.mplstyle'):

        fig, ax = plt.subplots()

        box_plot_data_indexes = (0, 3)

        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(box_plot_data[0][3], box_plot_data[1][3])
        test_str = f'Test Mann-Whitney U : p < 0.001'

        # Sample sizes
        n1 = box_plot_data[0][4]
        n2 = box_plot_data[3][4]
        n_total = n1 + n2

        ax.boxplot([box_plot_data[i][3] for i in box_plot_data_indexes])

        ax.yaxis.grid(True, linestyle='--', alpha=0.625)

        ax.set_xticklabels((f'Reprise (n = {n1})', f'Pas reprise (n = {n2})'))
        plt.title(f'Relation entre la reprise d\'un cours et l\'abandon (n = {n_total})')
        plt.ylabel('Taux d\'abandon [%]')

        plt.show()


    with plt.style.context('./images/main_plot_style.mplstyle'):
        box_plot_data_indexes = (71, 74, 77)
        plt.boxplot([box_plot_data[i][3] for i in box_plot_data_indexes])
        plt.gca().set_xticklabels(('Aisée', 'Satisfaisante', 'Précaire'))
        plt.title('Influence de la situation financière auto-déclarée')
        plt.ylabel('Taux d\'échec [%]')
        plt.show()

    pass