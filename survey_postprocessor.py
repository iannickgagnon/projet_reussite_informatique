
# External libraries
import pickle
import numpy as np

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

    questions_and_outcomes = Course.compile_survey_results_from_courses(courses)

    NUMERICAL_QUESTIONS_INDEX = (6,)

    # Analyser tous les sigles confondus
    is_found = 0
    for course in courses:
        for answers in course.surveys:
            for student in course.students:
                if student.name == answers.student_name:
                    for question_index, answer in enumerate(answers):
                        if question_index in NUMERICAL_QUESTIONS_INDEX:
                            questions_and_outcomes[question_index].append(answers[question_index])
                        else:
                            questions_and_outcomes[question_index][answer][student.get_outcome()] += 1

    # Count the total number of answers
    total_counts = sum((sum(d.values()) for d in questions_and_outcomes[0].values()))

    # Parse answers
    for question_index, question in enumerate(questions_and_outcomes.values()):

        # No confidence intervals for numerical questions
        if question_index in NUMERICAL_QUESTIONS_INDEX:
            continue

        # Show question number
        print(f'Question no.{question_index + 1}\n')

        # Generate bins from question dictionary
        bins_question = [sum(d.values()) for d in question.values()]

        # Calculate the number of bins
        nb_bins = len(bins_question)

        # Generate bootstrap samples
        bootstrap_samples_answers = bootstrap_generate_samples_from_bins(bins_question, total_counts, nb_bins)


        answer_index = 1
        for answer_key, value, sample in zip(question.keys(), bins_question, bootstrap_samples_answers):

            # Calculate confidence interval for current question
            lower_bound, upper_bound = bootstrap_calculate_confidence_interval(sample)

            # Transform in percentages
            value = value / total_counts * 100
            lower_bound = lower_bound / total_counts * 100
            upper_bound = upper_bound / total_counts * 100

            # Show proportions for each answer
            print(f'\tAnswer {answer_index} -> \'{answer_key}\' : {value:<5.1f}% [{lower_bound:5.1f}, {upper_bound:5.1f}]\n')

            # Parse outcomes
            outcomes = questions_and_outcomes[question_index][answer_key]

            # Generate bins from question dictionnary
            bins_outcomes = list(outcomes.values())
            total_outcomes = sum(bins_outcomes)

            # Generate bootstrap samples
            bootstrap_samples_outcomes = bootstrap_generate_samples_from_bins(bins_outcomes, sum(bins_outcomes), COURSE_NB_OUTCOMES)

            outcome_index = 1
            for outcome_key, value, sample_outcome in zip(COURSE_OUTCOMES, bins_outcomes, bootstrap_samples_outcomes):

                # Calculate confidence interval for current question
                lower_bound, upper_bound = bootstrap_calculate_confidence_interval(sample_outcome)

                # Transform in percentages
                value = value / total_outcomes * 100
                lower_bound = lower_bound / total_outcomes * 100
                upper_bound = upper_bound / total_outcomes * 100

                # Show proportions for each answer
                confidence_interval_str = confidence_interval_to_string(value, lower_bound, upper_bound)
                print(f'\t\tOutcome {outcome_index} -> \'{outcome_key:7s}\' : {value:<5.1f}% [{lower_bound:5.1f}, {upper_bound:5.1f}]\t{confidence_interval_str}')


                outcome_index += 1

            answer_index += 1

            print()