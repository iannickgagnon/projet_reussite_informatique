
# External librearies
import pandas as pd
import pytest

# Internal libraries
from survey import Survey

from anonymizer import (
    anonymize,
    __clean_up,
)

from test_results_parser import test_data_expected_grades
from test_events_parser import test_data_expected_events
from test_survey import test_data_survey

from events_parser import parse_events

from constants import PATH_TEST_EVENTS_FROM_TESTS

def test_anonymize(test_data_expected_grades,
                   test_data_expected_events,
                   test_data_survey):

    course_id = 'INF136'
    semester = 'A2022'

    events = parse_events(PATH_TEST_EVENTS_FROM_TESTS)
    results = test_data_expected_grades
    surveys = test_data_survey

    events_clean, results_clean, surveys_clean = anonymize(course_id,
                                                           semester,
                                                           events,
                                                           results,
                                                           surveys,
                                                           clean_up=True)

    assert True
