
# External libraries
import pickle
import graphviz
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from tools import plot_confidence_intervals
from predictor import run_model_and_evaluate
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
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
    _0_EVENTS,
    _0_PERCENT,
    _50_PERCENT,
    _100_PERCENT,
    PLOT_BORDER_MARGIN_FACTOR_SMALL,
    PLOT_X_LABEL_ENGAGEMENT,
    PASSING_GRADE,
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

    # Show all
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


def analysis_8():
    """
    Compiles survey answers, calculates confidence intervals, prints formatted results and plots selected results.
    """

    # Load data
    with open('courses.pkl', 'rb') as file:
        courses = pickle.load(file)

    # Compile survey results
    compiled_survey_results = Course.compile_survey_results(courses)

    # Count the total number of answers
    total_counts = sum((sum(d.values()) for d in compiled_survey_results[0].values()))

    # Initialize data structure for box plots
    box_plot_data = []

    # Parse answers
    for question_index, question in enumerate(compiled_survey_results.values()):

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
            value = value / total_counts * _100_PERCENT
            lower = lower / total_counts * _100_PERCENT
            upper = upper / total_counts * _100_PERCENT

            # Show proportions for each answer
            print(f'\tANSWER \'{answer_key}\': {value:<4.1f}% [{lower:4.1f}, {upper:4.1f}]\n')

            # Parse outcomes
            outcomes = compiled_survey_results[question_index][answer_key]

            # Generate bins from question dictionary
            bins_outcomes = list(outcomes.values())
            total_outcomes = sum(bins_outcomes)

            # Generate bootstrap samples
            bootstrap_outcomes = bootstrap_generate_samples_from_bins(bins_outcomes,
                                                                      sum(bins_outcomes),
                                                                      COURSE_NB_OUTCOMES)

            # Calculate confidence intervals on bootstrap outcomes
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
                value_outcome = value_outcome / total_outcomes * _100_PERCENT
                lower = lower / total_outcomes * _100_PERCENT
                upper = upper / total_outcomes * _100_PERCENT

                # Show proportions for each answer
                print(f'\t\t\'{outcome_key:7s}\': {value_outcome:<4.1f}% [{lower:4.1f}, {upper:4.1f}]\t')

            print()

    # Generate plots
    with plt.style.context(PATH_MAIN_PLOT_STYLE):

        '''
            Financial situation
        '''

        # Create plot
        _, ax1 = plt.subplots()

        # Corresponding indexes in boxplot data structure
        box_plot_data_indexes = (71, 74, 77)

        # Sample sizes
        n1 = box_plot_data[box_plot_data_indexes[0]][4]
        n2 = box_plot_data[box_plot_data_indexes[1]][4]
        n3 = box_plot_data[box_plot_data_indexes[2]][4]
        n_total = n1 + n2 + n3

        # Create boxplot
        ax1.boxplot([box_plot_data[i][3] for i in box_plot_data_indexes])

        # Add y-axis grid
        ax1.yaxis.grid(True, linestyle='--', alpha=0.625)

        # Add answer labels with sample sizes
        ax1.set_xticklabels((f'Aisée (n = {n1})', f'Satisfaisante (n = {n2})', f'Précaire (n = {n3})'))

        # Decorate
        plt.title(f'Relation entre la situation financière et l\'échec (n = {n_total})')
        plt.ylabel('Taux d\'échec [%]')

        '''
            Having children or not
        '''

        # Create plot
        _, ax2 = plt.subplots()

        # Corresponding indexes in boxplot data structure
        box_plot_data_indexes = (57, 60)

        # Sample sizes
        n1 = box_plot_data[box_plot_data_indexes[0]][4]
        n2 = box_plot_data[box_plot_data_indexes[1]][4]
        n_total = n1 + n2

        # Create boxplot
        ax2.boxplot([box_plot_data[i][3] for i in box_plot_data_indexes])

        # Add y-axis grid
        ax2.yaxis.grid(True, linestyle='--', alpha=0.625)

        # Add answer labels with sample sizes
        ax2.set_xticklabels((f'Enfants (n = {n1})', f'Pas d\'enfants (n = {n2})'))

        # Decorate
        plt.title(f'Relation entre le fait d\'avoir des enfants et l\'abandon (n = {n_total})')
        plt.ylabel('Taux d\'abandon [%]')

        '''
            Language spoken at home
        '''

        # Create plot
        _, ax3 = plt.subplots()

        # Corresponding indexes in boxplot data structure
        box_plot_data_indexes = (48, 54)

        # Sample sizes
        n1 = box_plot_data[box_plot_data_indexes[0]][4]
        n2 = box_plot_data[box_plot_data_indexes[1]][4]
        n_total = n1 + n2

        # Create boxplot
        ax3.boxplot([box_plot_data[i][3] for i in box_plot_data_indexes])

        # Add y-axis grid
        ax3.yaxis.grid(True, linestyle='--', alpha=0.625)

        # Add answer labels with sample sizes
        ax3.set_xticklabels((f'Français (n = {n1})', f'Autre (n = {n2})'))

        # Decorate
        plt.title(f'Relation entre la langue parlée à la maison et l\'abandon (n = {n_total})')
        plt.ylabel('Taux d\'abandon [%]')

        '''
            Mother tongue
        '''

        # Create plot
        _, ax4 = plt.subplots()

        # Corresponding indexes in boxplot data structure
        box_plot_data_indexes = (39, 45)

        # Sample sizes
        n1 = box_plot_data[box_plot_data_indexes[0]][4]
        n2 = box_plot_data[box_plot_data_indexes[1]][4]
        n_total = n1 + n2

        # Create boxplots
        ax4.boxplot([box_plot_data[i][3] for i in box_plot_data_indexes])

        # Add y-axis grid
        ax4.yaxis.grid(True, linestyle='--', alpha=0.625)

        # Add answer labels with sample sizes
        ax4.set_xticklabels((f'Français (n = {n1})', f'Autre (n = {n2})'))

        # Decorate
        plt.title(f'Relation entre la langue maternelle et l\'abandon (n = {n_total})')
        plt.ylabel('Taux d\'abandon [%]')

        '''
            Work and study situation
        '''

        # Create plot
        _, ax5 = plt.subplots()

        # Corresponding indexes in boxplot data structure
        box_plot_data_indexes = (32, 38)

        # Sample sizes
        n1 = box_plot_data[box_plot_data_indexes[0]][4]
        n2 = box_plot_data[box_plot_data_indexes[1]][4]
        n_total = n1 + n2

        # Create boxplots
        ax5.boxplot([box_plot_data[i][3] for i in box_plot_data_indexes])

        # Add y-axis grid
        ax5.yaxis.grid(True, linestyle='--', alpha=0.625)

        # Add answer labels with sample sizes
        ax5.set_xticklabels((f'Études (n = {n1})', f'Études + travail (n = {n2})'))

        # Decorate
        plt.title(f'Relation entre le travail et l\'échec (n = {n_total})')
        plt.ylabel('Taux d\'échec [%]')

        '''
            Retaking the course and failure
        '''

        # Create plot
        _, ax6 = plt.subplots()

        # Corresponding indexes in boxplot data structure
        box_plot_data_indexes = (2, 5)

        # Sample sizes
        n1 = box_plot_data[box_plot_data_indexes[0]][4]
        n2 = box_plot_data[box_plot_data_indexes[1]][4]
        n_total = n1 + n2

        # Create boxplots
        ax6.boxplot([box_plot_data[i][3] for i in box_plot_data_indexes])

        # Add y-axis grid
        ax6.yaxis.grid(True, linestyle='--', alpha=0.625)

        # Add answer labels with sample sizes
        ax6.set_xticklabels((f'Reprise (n = {n1})', f'Pas reprise (n = {n2})'))

        # Decorate
        plt.title(f'Relation entre la reprise d\'un cours et l\'échec (n = {n_total})')
        plt.ylabel('Taux d\'échec [%]')

        '''
            Retaking the course and dropping
        '''

        # Create plot
        _, ax7 = plt.subplots()

        # Corresponding indexes in boxplot data structure
        box_plot_data_indexes = (0, 3)

        # Sample sizes
        n1 = box_plot_data[0][4]
        n2 = box_plot_data[3][4]
        n_total = n1 + n2

        # Create boxplots
        ax7.boxplot([box_plot_data[i][3] for i in box_plot_data_indexes])

        # Add y-axis grid
        ax7.yaxis.grid(True, linestyle='--', alpha=0.625)

        # Add answer labels with sample sizes
        ax7.set_xticklabels((f'Reprise (n = {n1})', f'Pas reprise (n = {n2})'))

        # Decorate
        plt.title(f'Relation entre la reprise d\'un cours et l\'abandon (n = {n_total})')
        plt.ylabel('Taux d\'abandon [%]')

    # Show all
    plt.show()


def analysis_9():
    """
    Plots a list of courses' individual averages against quadratic delay.
    """

    is_plot_successes = True
    is_plot_failures = True
    is_linear_regression = True

    # TODO: Remove
    with open('courses.pkl', 'rb') as file:
        courses = pickle.load(file)

    quadratic_delay_pass = []
    course_delays_pass = []
    quadratic_delay_fail = []
    course_delays_fail = []

    # Split failures and successes
    for course in courses:

        # Get events counts and individual averages
        course_averages = course.get_individual_avg_vector()
        course_delays = course.get_quadratic_delay_vector()

        for i in range(len(course_averages)):

            # Separate failures from successes
            if course_averages[i] >= PASSING_GRADE:
                quadratic_delay_pass.append(course_averages[i])
                course_delays_pass.append(course_delays[i])
            else:
                quadratic_delay_fail.append(course_averages[i])
                course_delays_fail.append(course_delays[i])

    # Plot
    with plt.style.context(PATH_MAIN_PLOT_STYLE):

        # Create figure and axes
        fig, ax = plt.subplots()

        # Plot successes
        if is_plot_successes:
            ax.scatter(course_delays_pass, quadratic_delay_pass, color='royalblue', s=4)

        # Plot failures
        if is_plot_failures:
            ax.scatter(course_delays_fail, quadratic_delay_fail, color='indianred', s=4)

        # Build data vectors based on regression options
        if is_plot_successes and is_plot_failures:
            # Combine failures and successes
            course_delays_fail.extend(course_delays_pass)
            quadratic_delay_fail.extend(quadratic_delay_pass)
            data_x = np.array(course_delays_fail).reshape(-1, 1)
            data_y = np.array(quadratic_delay_fail).reshape(-1, 1)
        elif is_plot_successes:
            data_x = np.array(course_delays_pass).reshape(-1, 1)
            data_y = np.array(quadratic_delay_pass).reshape(-1, 1)
        else:
            data_x = np.array(course_delays_fail).reshape(-1, 1)
            data_y = np.array(quadratic_delay_fail).reshape(-1, 1)

        # Linear regression for failures
        if is_linear_regression:
            # Create model
            model = LinearRegression()
            model.loss = 'mae'

            # Fit model to data
            model.fit(data_x, data_y)

            # Find min/max for x-axis
            x_min = course_delays_pass[np.argmin(course_delays_pass)]
            x_max = course_delays_pass[np.argmax(course_delays_pass)]

            # Model output
            x_regression = np.linspace(x_min, x_max, 100)
            y_regression = model.predict(x_regression[:, np.newaxis])

            # Add to plot
            ax.plot(x_regression, y_regression, 'k--')

        # Labels
        plt.xlabel('Quadratic delay [min^2]')
        plt.ylabel('Individual average [%]')

        # Limits
        plt.xlim(_0_EVENTS, max(course_delays_pass + course_delays_fail) * PLOT_BORDER_MARGIN_FACTOR_SMALL)
        plt.ylim(0, 100 * PLOT_BORDER_MARGIN_FACTOR_SMALL)

        if is_plot_successes and is_plot_failures:
            legend_items = ['Pass', 'Fail']
        elif is_plot_successes:
            legend_items = ['Pass']
        else:
            legend_items = ['Fail']

        # Specify MAE regression in title
        if is_linear_regression:
            plt.title('Linear regression using Mean Absolute Error')

        # Add legend
        plt.legend(legend_items, loc='upper left')

    # Show plot
    plt.show()

    # Print coefficients if regression is done
    if is_linear_regression:
        print(f'Slope     : {model.coef_[0][0]:.2f}')
        print(f'Intercept : {model.intercept_[0]:.2f}')

    with plt.style.context('./images/main_plot_style.mplstyle'):
        plt.title('Relation entre le délai d\'engagement quadratique moyen et le résultat (n=719)')
        plt.ylabel('Délai d\'engagement quadratique moyen [min^2]')
        plt.legend(('Succès', 'Échec'))

    # Export figure and axes
    return fig, ax


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
    analysis_7()
    analysis_8()

    '''

    analysis_9()

    '''
    # Rebuild and save
    courses = Course.build_course_list_from_files()
    with open('courses.pkl', 'wb') as file:
        pickle.dump(courses, file)
    '''

