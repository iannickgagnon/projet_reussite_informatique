
# External libraries
import pickle
import graphviz
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from tools import plot_confidence_intervals
from predictor import run_model_and_evaluate
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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


if __name__ == '__main__':

    '''
    analysis_1_a('INF135_02.csv')
    analysis_1_b('INF135_02.csv')
    analysis_1_c('INF135_02.csv')
    analysis_2()
    analysis_3()
    analysis_4()
    analysis_5()
    
    '''

    analysis_6()

    '''
    # Rebuild and save
    courses = Course.build_course_list_from_files()
    with open('courses.pkl', 'wb') as file:
        pickle.dump(courses, file)à
    '''
