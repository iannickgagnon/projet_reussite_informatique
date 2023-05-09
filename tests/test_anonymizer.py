# External libraries
import pandas as pd

# Internal libraries
from anonymizer import anonymize

# Pytest fixtures
from test_results_parser import test_data_expected_grades
from test_events_parser import test_data_expected_events
from test_survey import test_data_expected_survey

# Internal constants
from constants import SURVEY_NB_QUESTIONS


def test_anonymize(test_data_expected_grades,
                   test_data_expected_events,
                   test_data_expected_survey):

    # Expected outputs
    expected_events = pd.DataFrame({
        "Name": ['INF136_A2022_0', 'INF136_A2022_1'],
        "Event": ['Journal consulté', 'Cours consulté'],
        "Date": ['2022-01-1', '2022-01-1'],
        "Time": [9.0, 10.5]
    })

    expected_results = pd.DataFrame({
        "Name": ['INF136_A2022_0', 'INF136_A2022_1'],
        "EXAM01": [100.0, 50.0],
        "EXAM02": [100.0, 50.0],
        "TP01": [100.0, 50.0],
        "FINAL": [100.0, 50.0],
        "TP02": [100.0, 50.0],
        "RAP01": [100.0, 50.0],
        "TP03": [100.0, 50.0],
        "Average": [100.0, 50.0],
        "Grade": ['A+', 'C']
    })

    # Test data
    course_id = 'INF136'
    semester = 'A2022'
    events = test_data_expected_events
    results = test_data_expected_grades
    surveys = test_data_expected_survey

    events_clean, results_clean, surveys_clean = anonymize(course_id,
                                                           semester,
                                                           events,
                                                           results,
                                                           surveys,
                                                           clean_up=True)

    # Validate cleaned up anonymized events
    assert events_clean.equals(expected_events)

    # Validate cleaned up anonymized results
    assert results_clean.equals(expected_results)

    # Validate cleaned up anonymized surveys
    assert surveys.count == 2

    assert surveys[0].answers == {0: 'Non', 1: 'Non', 2: 'Non', 3: 'Non', 4: 'Non', 5: 'Études seulement', 6: 10,
                                  7: 'Français', 8: 'Français', 9: 'Non', 10: 'Non', 11: 'Aisée'}
    assert surveys[0].nb_questions == SURVEY_NB_QUESTIONS
    assert surveys[0].student_name == 'INF136_A2022_0'

    assert surveys[1].answers == {0: 'Oui', 1: 'Oui', 2: 'Oui', 3: 'Oui', 4: 'Oui', 5: 'Travail seulement', 6: 20,
                                  7: 'Autre', 8: 'Autre', 9: 'Oui', 10: 'Oui', 11: 'Satisfaisante'}
    assert surveys[1].nb_questions == SURVEY_NB_QUESTIONS
    assert surveys[1].student_name == 'INF136_A2022_1'
