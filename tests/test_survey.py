
# External libraries
import pytest

# Internal libraries
from survey import Survey
from survey_answers import SurveyAnswers

# Internal constants
from constants import (
    PATH_TEST_SURVEY_FROM_TESTS,
    SURVEY_NB_QUESTIONS,
)


@pytest.fixture
def survey_data():

    # Generate survey from test data
    test_survey = Survey(filename=PATH_TEST_SURVEY_FROM_TESTS)

    return test_survey


def test_survey_build():

    # Generate test survey manually
    survey = Survey(filename=PATH_TEST_SURVEY_FROM_TESTS)

    # Validate the count property
    assert survey.count == 2

    # Validate the number of stored answer sets
    assert len(survey.answer_sets) == 2

    # Validate first answer set
    answer_set_1 = survey.answer_sets[0]

    # Validate properties
    assert answer_set_1.nb_questions == SURVEY_NB_QUESTIONS
    assert answer_set_1.student_name == 'John Doe'

    # Validate answers
    for i in (0, 1, 2, 3, 4, 9, 10):
        assert answer_set_1.answers[i] == 'Non'

    assert answer_set_1.answers[5] == 'Études seulement'
    assert answer_set_1.answers[6] == 10
    assert answer_set_1.answers[7] == 'Français'
    assert answer_set_1.answers[8] == 'Français'
    assert answer_set_1.answers[11] == 'Aisée'

    # Validate second answer set
    answer_set_2 = survey.answer_sets[1]

    # Validate properties
    assert answer_set_2.nb_questions == SURVEY_NB_QUESTIONS
    assert answer_set_2.student_name == 'Jane Doe'

    # Validate answers
    for i in (0, 1, 2, 3, 4, 9, 10):
        assert answer_set_2.answers[i] == 'Oui'

    assert answer_set_2.answers[5] == 'Travail seulement'
    assert answer_set_2.answers[6] == 20
    assert answer_set_2.answers[7] == 'Autre'
    assert answer_set_2.answers[8] == 'Autre'
    assert answer_set_2.answers[11] == 'Satisfaisante'


def test_survey_get_item(survey_data):

    # Test access operator implementation
    assert isinstance(survey_data[0], SurveyAnswers)


def test_survey_del_item(survey_data):

    # Delete the first answer set
    del survey_data[0]

    # Check that the right one was deleted
    assert survey_data[0].student_name == 'Jane Doe'

    # Check that there is only one out of two answer sets left
    assert len(survey_data) == 1


def test_survey_iterator(survey_data):

    # Expected names
    expected_names = ('John Doe', 'Jane Doe')

    # Check that iteration works by looking at the student_name property of the returned elements
    for answer, expected_name in zip(survey_data, expected_names):
        assert answer.student_name == expected_name


def test_survey_filter_by_student_name(survey_data):

    # Filter surveys
    filtered_survey_1 = survey_data.filter_by_student_name('John Doe')
    filtered_survey_2 = survey_data.filter_by_student_name('Jane Doe')

    # Check that a single survey was returned for each
    assert len(filtered_survey_1) == 1
    assert len(filtered_survey_2) == 1

    # Check that the appropriate surveys were returned
    assert filtered_survey_1[0].student_name == 'John Doe'
    assert filtered_survey_2[0].student_name == 'Jane Doe'
