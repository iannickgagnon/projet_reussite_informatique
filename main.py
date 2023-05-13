
# External libraries
import pickle
from copy import deepcopy
from statistics import mean
import matplotlib.pyplot as plt

# Internal libraries
from survey import Survey
from course import Course
from anonymizer import anonymize
from events_parser import parse_events
from results_parser import parse_results


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

    # TODO: Remove
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


if __name__ == '__main__':

    #analysis_1_a('INF135_02.csv')
    #analysis_1_b('INF135_02.csv')
    #analysis_1_c('INF135_02.csv')
    analysis_2()
    #analysis_3()
    #analysis_4()

    """
    # Rebuild and save
    courses = Course.build_course_list_from_files()
    with open('courses.pkl', 'wb') as file:
        pickle.dump(courses, file)à
    """