
# External libraries
import numpy as np
import pandas as pd

# Internal libraries
from survey import Survey
from course import Course
from anonymizer import anonymize

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
    with open('courses.pkl', 'rb') as file:
        courses = pickle.load(file)

    course_dataset = []
    for current_course in courses:
        for current_student in current_course.students:
            student_outcome = current_student.get_outcome()
            current_student_survey = Survey.filter_by_student_name(current_course.surveys, current_student.name)
            course_dataset.append([answer for answer in current_student_survey[0]] + [student_outcome])

    # Convert to DataFrame
    course_dataset = pd.DataFrame(course_dataset, columns=[f'Q{i}' for i in range(1, SURVEY_NB_QUESTIONS + 1)] + ['Outcome'])

    # Encode discrete input variables
    encoded_course_dataset = pd.get_dummies(course_dataset)

    # Multiple linear regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    # Split into predictors and output
    x_data = encoded_course_dataset.iloc[:, :-1]
    y_data = encoded_course_dataset.iloc[:, -1]

    # Split into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model and fit to data
    model = LogisticRegression()
    model.fit(x_data, y_data)

    # Make predictions using the trained model
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calculate performance metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Show performance metrics
    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("Training Precision:", train_precision)
    print("Test Precision:", test_precision)
    print("Training Recall:", train_recall)
    print("Test Recall:", test_recall)
    print("Training F1 Score:", train_f1)
    print("Test F1 Score:", test_f1)

    # Repeat for decision trees
    print('\nDECISION TREES')

    model_decision_tree = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_train_pred_tree = model.predict(x_train)
    y_test_pred_tree = model.predict(x_test)

    # Calculate performance metrics
    train_accuracy_tree = accuracy_score(y_train, y_train_pred)
    test_accuracy_tree = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Print the performance metrics
    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("Training Precision:", train_precision)
    print("Test Precision:", test_precision)
    print("Training Recall:", train_recall)
    print("Test Recall:", test_recall)
    print("Training F1 Score:", train_f1)
    print("Test F1 Score:", test_f1)

    from sklearn.svm import SVC
    model = SVC(kernel='rbf')
    model.fit(x_train, y_train)

    """
    import pandas as pd
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Load the dataframe
    df = pd.read_csv('your_dataframe.csv')
    
    # Split the dataframe into input features (X) and the output variable (y)
    X = df.iloc[:, :-1]  # Select all columns except the last one as input features
    y = df.iloc[:, -1]   # Select the last column as the output variable
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the parameter grid for the grid search
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf'],
    }
    
    # Initialize and fit the SVM classifier with grid search
    model = SVC()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Get the best model from the grid search
    best_model = grid_search.best_estimator_
    
    # Predict using the best model
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Calculate performance metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # Print the performance metrics
    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("Training Precision:", train_precision)
    print("Test Precision:", test_precision)
    print("Training Recall:", train_recall)
    print("Test Recall:", test_recall)
    print("Training F1 Score:", train_f1)
    print("Test F1 Score:", test_f1)
    
    # Print the best hyperparameters found by grid search
    print("Best Hyperparameters:", grid_search.best_params_)
    """

    print()
