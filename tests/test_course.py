
# External libraries
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Internal libraries
from survey import Survey
from course import Course
from anonymizer import anonymize

from predictor import train_and_test_model
from predictor import calculate_performance_metrics

# Pytest fixtures
from test_events_parser import test_data_expected_events
from test_results_parser import test_data_expected_structure
from test_results_parser import test_data_expected_grades
from test_survey import test_data_expected_survey

# Internal constants
from constants import SURVEY_NB_QUESTIONS


def test_to_dataset(test_data_expected_structure,
                    test_data_expected_events,
                    test_data_expected_grades,
                    test_data_expected_survey):

    """
    # Test data
    course_id = 'INF136'
    semester_id = 'A2022'

    # Load test data
    evaluation_structure = test_data_expected_structure
    events = test_data_expected_events
    results = test_data_expected_grades
    surveys = test_data_expected_survey

    # TODO: If not anonymized, the incomplete student break the code
    events_clean, results_clean, surveys_clean = anonymize(course_id,
                                                           semester_id,
                                                           events,
                                                           results,
                                                           surveys,
                                                           clean_up=True)

    # Build course
    course_obj = Course(evaluation_structure,
                        course_id,
                        semester_id,
                        events_clean,
                        results_clean,
                        surveys_clean)
    """

    #TODO: Refactor in Student.to_vector() method
    import pickle
    with open('..\courses.pkl', 'rb') as file:
        courses = pickle.load(file)

    course_dataset = []
    for current_course in courses:
        for current_student in current_course.students:
            student_outcome = current_student.get_outcome()
            current_student_survey = Survey.filter_by_student_name(current_course.surveys, current_student.name)

            # TODO: This is necessary because the clean_up option wasn't used when the data was anonymized
            if current_student_survey is not None:
                course_dataset.append([answer for answer in current_student_survey[0]] + [current_student.nb_events, student_outcome])

    # Convert to DataFrame
    course_dataset = pd.DataFrame(course_dataset, columns=[f'Q{i}' for i in range(1, SURVEY_NB_QUESTIONS + 1)] + ['Events', 'Outcome'])

    # Encode predictors and response
    x_data = pd.get_dummies(course_dataset.iloc[:, :-1])
    y_encoder = LabelEncoder()
    y_data = pd.DataFrame({'Outcome': y_encoder.fit_transform(course_dataset.iloc[:, -1])})

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
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

    for model_name, model in models.items():

        # Train and test the model
        y_train_pred, y_test_pred = train_and_test_model(model, x_train, y_train, x_test)

        # Show model name
        print(f'\n\n{model_name}\n')

        # Evaluate model and show
        calculate_performance_metrics(y_train, y_train_pred, y_test, y_test_pred, labels=y_encoder.classes_)
