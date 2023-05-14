
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

    #TODO: Refactor in Student.to_vector() method
    course_dataset = []
    for current_student in course_obj.students:
        student_outcome = current_student.get_outcome()
        current_student_survey = Survey.filter_by_student_name(surveys, current_student.name)
        course_dataset.append([answer for answer in current_student_survey[0]] + [student_outcome])

    # Convert to DataFrame
    course_dataset = pd.DataFrame(course_dataset, columns=[f'Q{i}' for i in range(1, SURVEY_NB_QUESTIONS + 1)] + ['Outcome'])



    # TODO: Encode strings

    """
    import pandas as pd
    
    # Example DataFrame
    data = pd.DataFrame({'Color': ['red', 'blue', 'green', 'red', 'blue']})
    
    # Perform one-hot encoding
    encoded_data = pd.get_dummies(data['Color'])
    
    # Concatenate the encoded data with the original DataFrame
    data_encoded = pd.concat([data, encoded_data], axis=1)
    
    # Print the encoded data
    print(data_encoded)
    """

    print()
