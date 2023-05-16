
# External libraries
import pickle
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Internal libraries
from survey import Survey
from course import Course
from anonymizer import anonymize
from events_parser import parse_events
from results_parser import parse_results
from predictor import train_and_test_model
from predictor import calculate_performance_metrics

# Internal constants
from constants import SURVEY_NB_QUESTIONS


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


def analysis_5():
    """
    Plots the stacked histograms of failures and successes distributions.
    """

    import pickle
    with open('courses.pkl', 'rb') as file:
        courses = pickle.load(file)

    # Extract data from list of courses
    course_data = Course.course_list_to_table(courses)

    # Encode predictors and response
    x_data = pd.get_dummies(course_data.iloc[:, :-1])

    # Response using one-hot encoding to get a vector
    y_encoder = LabelEncoder()
    y_data = pd.DataFrame({'Outcome': y_encoder.fit_transform(course_data.iloc[:, -1])})

    # Extract encoder classes
    classes = y_encoder.classes_

    # Balance classes using Synthetic Minority Oversampling TEchnique (SMOTE)
    smote = SMOTE()
    x_data_resampled, y_data_resampled = smote.fit_resample(x_data, y_data)

    '''
        MAKE PREDICTIONS
    '''

    # Initialize model
    models = {'Logistic regression': LogisticRegression(),
              'Decision trees': DecisionTreeClassifier(),
              'Random forest': RandomForestClassifier(),
              'SVM with linear kernel': SVC(kernel='linear'),
              'SVM with polynomial kernel': SVC(kernel='poly', degree=3),
              'SVM with radial RBF kernel': SVC(kernel='rbf')}

    # Split into training and test sets
    x_train_resampled, x_test_resampled, y_train_resampled, y_test_resampled = \
        train_test_split(x_data_resampled, y_data_resampled, test_size=0.2)

    # Run models and evaluate
    for model_name, model in models.items():

        # Train and test the model
        y_train_pred_resampled, y_test_pred_resampled = \
            train_and_test_model(model, x_train_resampled, y_train_resampled, x_test_resampled)

        # Show model name
        print(f'\n\nRESAMPLED {model_name}\n')

        # Evaluate model and show
        calculate_performance_metrics(y_train_resampled,
                                      y_train_pred_resampled,
                                      y_test_resampled,
                                      y_test_pred_resampled,
                                      labels=classes)

        # Calculate performance metrics on original dataset
        print(f'\n\nNOT RESAMPLED {model_name}\n')

        y_pred_not_resampled = model.predict(x_data)

        # Show performance metrics
        train_accuracy = accuracy_score(y_data, y_pred_not_resampled)
        train_precision = precision_score(y_data, y_pred_not_resampled, average=None)
        train_recall = recall_score(y_data, y_pred_not_resampled, average=None)
        train_f1 = f1_score(y_data, y_pred_not_resampled, average=None)

        precision_str = ' '.join([f'{label}: {precision:.2f}\t' for label, precision in zip(classes, train_precision)])
        recall_str = ' '.join([f'{label}: {recall:.2f}\t' for label, recall in zip(classes, train_recall)])
        f1_str = ' '.join([f'{label}: {f1:.2f}\t' for label, f1 in zip(classes, train_f1)])

        print("\tAccuracy  :", train_accuracy)
        print("\tPrecision :", precision_str)
        print("\tRecall    :", recall_str)
        print("\tF1 Score  :", f1_str)


if __name__ == '__main__':

    '''
    analysis_1_a('INF135_02.csv')
    analysis_1_b('INF135_02.csv')
    analysis_1_c('INF135_02.csv')
    analysis_2()
    analysis_3()
    analysis_4()
    '''
    analysis_5()

    '''
    # Rebuild and save
    courses = Course.build_course_list_from_files()
    with open('courses.pkl', 'wb') as file:
        pickle.dump(courses, file)à
    '''