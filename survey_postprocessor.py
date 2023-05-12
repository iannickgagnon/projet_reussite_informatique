
# External libraries
import pickle
import numpy as np
from copy import deepcopy

from typing import (
  Iterable,
  Tuple,
  List,
)

# Internal constants
from constants import (
    COURSE_OUTCOMES,
    COURSE_NB_OUTCOMES,
)


def bootstrap_calculate_confidence_interval(base_vector: np.array, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculates (1 - alpha) confidence intervals (CI) for a given dataset.

    Args:
      data (Iterable): Data to calculate the CI for.
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


def bootstrap_generate_base_vector_from_bins(bins: Iterable) -> np.array:

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
                                         nb_samples: int = 1000) -> np.array:
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
                                  bar_length: int = 100):

    if lower_bound == upper_bound:
       return ''

    # Determine position of value
    position_val = int(bar_length * value / (bar_length * 2) )
    position_lower_bound = int(bar_length * lower_bound / (bar_length * 2))
    position_upper_bound = int(bar_length * upper_bound / (bar_length * 2))

    positions = (position_val, position_lower_bound, position_upper_bound)
    symbols = ('*', '|', '|')

    confidence_interval_string = '-' * bar_length

    for position, symbol in zip(positions, symbols):
        confidence_interval_string = confidence_interval_string[:position] + symbol + confidence_interval_string[position + 1:]

    # Build CI string
    return '[' + confidence_interval_string + ']'


if __name__ == '__main__':

    with open('courses.pkl', 'rb') as file:
        courses = pickle.load(file)

    OUTCOMES = {outcome: 0 for outcome in ('Abandon', 'Succès', 'Échec')}

    YES_NO = {'Oui': OUTCOMES.copy(),
              'Non': OUTCOMES.copy()}

    EMPLOI_ETE = {'Études seulement': OUTCOMES.copy(),
                  'Travail seulement': OUTCOMES.copy(),
                  'Études et travail': OUTCOMES.copy()}

    LANGUE = {'Français': OUTCOMES.copy(),
              'Anglais': OUTCOMES.copy(),
              'Autre': OUTCOMES.copy()}

    SITUATION_FINANCIERE = {'Aisée': OUTCOMES.copy(),
                            'Satisfaisante': OUTCOMES.copy(),
                            'Précaire': OUTCOMES.copy()}

    questions_and_outcomes = {0: deepcopy(YES_NO),
                              1: deepcopy(YES_NO),
                              2: deepcopy(YES_NO),
                              3: deepcopy(YES_NO),
                              4: deepcopy(YES_NO),
                              5: deepcopy(EMPLOI_ETE),
                              6: [],
                              7: deepcopy(LANGUE),
                              8: deepcopy(LANGUE),
                              9: deepcopy(YES_NO),
                              10: deepcopy(YES_NO),
                              11: deepcopy(SITUATION_FINANCIERE)}

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

        # Generate bins from question dictionnary
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