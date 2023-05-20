
# External libraries
import pickle
import graphviz
import pandas as pd
from scipy import stats
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from tools import plot_confidence_intervals
from predictor import run_model_and_evaluate
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tools import bootstrap_generate_samples_from_bins
from tools import bootstrap_calculate_confidence_interval

# Internal libraries
from survey import Survey
from course import Course
from anonymizer import anonymize
from events_parser import parse_events
from predictor import encode_full_data
from results_parser import parse_results

# Internal constants
from constants import PATH_MAIN_PLOT_STYLE

from constants import (
    SURVEY_NB_QUESTIONS,
    SURVEY_NUMERICAL_QUESTIONS_INDEX,
    COURSE_NB_OUTCOMES,
    COURSE_OUTCOMES,
)


def analysis_1_a(filename: str,
                 is_anonymize: bool = False,
                 is_regression: bool = False):
    """
    Plots a given course's average grades against engagement with or without a regression line.
    """

    # Get events data
    events_data, course_id, semester_id = parse_events(filename)

    # Get evaluation structure and grades
    evaluation_structure, results_data = parse_results(filename)

    # Anonymize
    if is_anonymize:

        # Get surveys data
        surveys_data = Survey(filename)

        # Invoke anonymizer
        events_data, results_data, surveys_data = anonymize(course_id,
                                                            semester_id,
                                                            events_data,
                                                            results_data,
                                                            surveys_data,
                                                            clean_up=False)
    else:

        # Surveys data is not required if data is not anonymized
        surveys_data = None

    # Build course
    course_obj = Course(evaluation_structure,
                        course_id,
                        semester_id,
                        events_data,
                        results_data,
                        surveys_data)

    # Build and show plot
    fig, ax = course_obj.plot_individual_avg_vs_engagement(normalize=True,
                                                           linear_regression=is_regression)

    # Export figure and axes
    return fig, ax


def analysis_1_b(filename: str,
                 is_anonymize: bool = False):
    """
    Plots a given course's average grades distribution.
    """

    # Get events data
    events_data, course_id, semester_id = parse_events(filename)

    # Get evaluation structure and grades
    evaluation_structure, results_data = parse_results(filename)

    # Anonymize
    if is_anonymize:

        # Get surveys data
        surveys_data = Survey(filename)

        # Invoke anonymizer
        events_data, results_data, surveys_data = anonymize(course_id,
                                                            semester_id,
                                                            events_data,
                                                            results_data,
                                                            surveys_data,
                                                            clean_up=False)
    else:

        # Surveys data is not required if data is not anonymized
        surveys_data = None

    # Build course
    course_obj = Course(evaluation_structure,
                        course_id,
                        semester_id,
                        events_data,
                        results_data,
                        surveys_data)

    # Build and show plot
    fig, ax = course_obj.plot_individual_avg_distribution()

    # Export figure and axes
    return fig, ax


def analysis_1_c(filename: str,
                 is_anonymize: bool = False):
    """
    Plots a given course's average engagement distribution.
    """

    # Get events data
    events_data, course_id, semester_id = parse_events(filename)

    # Get evaluation structure and grades
    evaluation_structure, results_data = parse_results(filename)

    # Anonymize
    if is_anonymize:

        # Get surveys data
        surveys_data = Survey(filename)

        # Invoke anonymizer
        events_data, results_data, surveys_data = anonymize(course_id,
                                                            semester_id,
                                                            events_data,
                                                            results_data,
                                                            surveys_data,
                                                            clean_up=False)
    else:

        # Surveys data is not required if data is not anonymized
        surveys_data = None

    # Build course
    course_obj = Course(evaluation_structure,
                        course_id,
                        semester_id,
                        events_data,
                        results_data,
                        surveys_data)

    # Build and show plot
    fig, ax = course_obj.plot_engagement_distribution()

    # Export figure and axes
    return fig, ax


def analysis_2():
    """
    Plots a list of courses' individual averages against engagement.
    """

    fig, ax = Course.plot_combined_individual_avg_vs_engagement(is_linear_regression=False,
                                                                is_plot_successes=True,
                                                                is_plot_failures=True)

    with plt.style.context('./images/main_plot_style.mplstyle'):
        plt.title('Relation entre l\'engagement et le résultat (n=719)')
        plt.ylabel('Moyenne obtenue au cours [%]')
        plt.legend(('Succès', 'Échec'))

    # Export figure and axes
    return fig, ax


def analysis_3():
    """
    Plots the number of points missing to go from failure to success.
    """

    fig, ax = Course.plot_combined_points_to_pass_vs_engagement(is_linear_regression=True)

    # Export figure and axes
    return fig, ax


def analysis_4():
    """
    Plots the stacked histograms of failures and successes distributions.
    """

    fig, ax = Course.plot_combined_stacked_distributions_pass_fail()

    # Export figure and axes
    return fig, ax


def analysis_5(show_graph=False):
    """
    Evaluates Random forest model and displays results.
    """

    # Load data
    with open('courses.pkl', 'rb') as file:
        courses = pickle.load(file)

    # Extract data from list of courses
    course_data = Course.course_list_to_table(courses)

    # Keep only quiz answers by extracting its columns
    columns_to_keep = [f'Q{i}' for i in range(1, SURVEY_NB_QUESTIONS + 1)] + ['Outcome']
    course_data = course_data[columns_to_keep]

    # Encode data
    x_data, y_data, classes = encode_full_data(course_data)

    # Initialize Decision Tree model
    model = RandomForestClassifier()
    model_name = 'Random forest'

    # Run model and evaluate
    run_model_and_evaluate(model, model_name, x_data, y_data, classes, nb_bootstrap_samples=10)

    # Graphical representation
    if show_graph:

        dot_data = export_graphviz(model['Random forest'],
                                   out_file=None,
                                   feature_names=x_data.columns,
                                   class_names=classes,
                                   filled=True,
                                   rounded=True,
                                   special_characters=True)

        graph = graphviz.Source(dot_data)
        graph.render("decision_tree_visual")
        graph.view()


def analysis_6():
    """
    Plots confidence 95% intervals for the 3 different classes.
    """

    # Abandon class confidence intervals
    low = [42, 64, 67, 77, 80, 83]
    mid = [53, 79, 80, 91, 92, 94]
    high = [68, 93, 94, 100, 100, 100]

    with plt.style.context(PATH_MAIN_PLOT_STYLE):

        # Create
        ax1 = plot_confidence_intervals(low, mid, high, x_labels=['1A', '1B', '1C', '2', '3', '4'])

        # Adjust y-axis limits
        plt.ylim([40, 102.5])

        # Add y-axis grid
        ax1.yaxis.grid(True, linestyle='--', alpha=0.625)

        # Decorate
        plt.title('Intervalles de confiance 95% pour l\'abandon')
        plt.xlabel('Modèle')
        plt.ylabel('Précision [%]')

    # Échec class confidence intervals
    low = [43, 54, 52, 70, 71, 78]
    mid = [52, 67, 68, 84, 86, 90]
    high = [62, 81, 85, 86, 100, 100]

    with plt.style.context(PATH_MAIN_PLOT_STYLE):

        # Create
        ax2 = plot_confidence_intervals(low, mid, high, x_labels=['1A', '1B', '1C', '2', '3', '4'])

        # Adjust y-axis limits
        plt.ylim([40, 102.5])

        # Add y-axis grid
        ax2.yaxis.grid(True, linestyle='--', alpha=0.625)

        # Decorate
        plt.title('Intervalles de confiance 95% pour l\'échec')
        plt.xlabel('Modèle')
        plt.ylabel('Précision [%]')

    # Succès class confidence intervals
    low = [95, 95, 90, 95, 95, 96]
    mid = [98, 95, 93, 97, 97, 99]
    high = [100, 96, 95, 100, 100, 100]

    with plt.style.context(PATH_MAIN_PLOT_STYLE):

        # Create
        ax3 = plot_confidence_intervals(low, mid, high, x_labels=['1A', '1B', '1C', '2', '3', '4'])

        # Adjust y-axis limits
        plt.ylim([40, 102.5])

        # Add y-axis grid
        ax3.yaxis.grid(True, linestyle='--', alpha=0.625)

        # Decorate
        plt.title('Intervalles de confiance 95% pour le succès')
        plt.xlabel('Modèle')
        plt.ylabel('Précision [%]')

    # Show all three graphs
    plt.show()


def analysis_7():

    # Load data
    with open('courses.pkl', 'rb') as file:
        courses = pickle.load(file)

    # Get the number of hours worked and associated outcome
    nb_hours, nb_hours_outcome = Course.compile_hours_worked_outcome(courses)

    # Separate number of hours worked based on outcome
    nb_hours_success = [h for h, o in zip(nb_hours, nb_hours_outcome) if o == 'Succès' and h > 0]
    nb_hours_abandon = [h for h, o in zip(nb_hours, nb_hours_outcome) if o == 'Abandon' and h > 0]
    nb_hours_fail = [h for h, o in zip(nb_hours, nb_hours_outcome) if o == 'Échec' and h > 0]

    with plt.style.context(PATH_MAIN_PLOT_STYLE):

        # Create figure and axes
        fig, ax = plt.subplots()

        # Create boxplot
        ax.boxplot([nb_hours_success, nb_hours_abandon, nb_hours_fail])

        # Add outcome labels
        ax.set_xticklabels(['Succès', 'Abandon', 'Échec'])

        # Show individual data points
        ax.scatter([1] * len(nb_hours_success), nb_hours_success, color='white', edgecolor='k', alpha=0.5)
        ax.scatter([2] * len(nb_hours_abandon), nb_hours_abandon, color='white', edgecolor='k', alpha=0.5)
        ax.scatter([3] * len(nb_hours_fail), nb_hours_fail, color='white', edgecolor='k', alpha=0.5)

        # Add y-axis grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.625)

        # Decorate
        plt.ylabel('Nombre d\'heures')

        plt.show()


if __name__ == '__main__':

    '''
    analysis_1_a('INF135_02.csv')
    analysis_1_b('INF135_02.csv')
    analysis_1_c('INF135_02.csv')
    analysis_2()
    analysis_3()
    analysis_4()
    analysis_5()
    analysis_6()
    '''
    analysis_7()
    '''
    # Rebuild and save
    courses = Course.build_course_list_from_files()
    with open('courses.pkl', 'wb') as file:
        pickle.dump(courses, file)
    '''

"""
if __name__ != '__main__':

    # Load data
    with open('courses.pkl', 'rb') as file:
        courses = pickle.load(file)

    # Compile survey results
    nb_compiled_survey_results = Course.compile_survey_results(courses)

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
            print(f'\tANSWER \'{answer_key}\': {value:<4.1f}% [{lower:4.1f}, {upper:4.1f}]\n')

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
                print(f'\t\t\'{outcome_key:7s}\': {value_outcome:<4.1f}% [{lower:4.1f}, {upper:4.1f}]\t')

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
"""
