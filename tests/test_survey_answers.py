# External libraries
import pytest

# Internal libraries
from survey_answers import SurveyAnswers


@pytest.fixture
def survey_answers():

    # Test data

    answers = {
        0: 'answer 1',
        1: 'answer 2',
        2: 'answer 3'
    }

    return SurveyAnswers(answers, 'Jane Doe')


def test_survey_answers_iter(survey_answers):

    # Generate iterator from class instance
    iterator = iter(survey_answers)

    # Test the next(...) implementation through __iter__()
    assert next(iterator) == 'answer 1'
    assert next(iterator) == 'answer 2'
    assert next(iterator) == 'answer 3'

    # Test iteration stopping mechanism through StopIteration exception
    with pytest.raises(StopIteration):
        next(iterator)


def test_survey_answers_getitem(survey_answers):

    # Test access operator implementation
    assert survey_answers[0] == 'answer 1'
    assert survey_answers[1] == 'answer 2'
    assert survey_answers[2] == 'answer 3'

    # Test index overflow exception
    with pytest.raises(IndexError):
        _ = survey_answers[3]


def test_survey_answers_repr(survey_answers):

    # Expected string
    expected = "SurveyAnswers(Jane Doe, {0: 'answer 1', ..., 2: 'answer 3'})"

    assert repr(survey_answers) == expected
