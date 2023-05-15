
# Pytest fixtures
from test_events_parser import test_data_expected_events
from test_results_parser import test_data_expected_structure
from test_results_parser import test_data_expected_grades
from test_survey import test_data_expected_survey


def test_course(test_data_expected_structure,
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