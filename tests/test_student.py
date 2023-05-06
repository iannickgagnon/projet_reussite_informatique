
# External libraries
import pytest
import pandas as pd

# Internal libraries
from student import Student

# Internal constants
from constants import (
    COL_NAME_AVERAGE,
    COL_NAME_GRADE,
    COL_NAME_EXAM,
    COL_NAME_FINAL,
    OUTCOME_ECHEC,
    OUTCOME_ABANDON,
    OUTCOME_SUCCES,
    OUTCOME_GRADE_ECHEC,
    OUTCOME_GRADES_ABANDON,
    LETTER_GRADE_B,
)


@pytest.fixture
def sample_student_data():

    # Mock name
    name = 'Iannick Gagnon'

    # Mock events
    events = pd.DataFrame({'event': ['event1', 'event2']})

    # Mock results
    results = pd.DataFrame({COL_NAME_AVERAGE: [80],
                            COL_NAME_GRADE: [LETTER_GRADE_B],
                            COL_NAME_EXAM + '1': [90],
                            COL_NAME_EXAM + '2': [85],
                            COL_NAME_FINAL: [95]})

    return name, events, results


def test_student_init(sample_student_data):

    # Get test data
    name, events, results = sample_student_data

    # Generate test student
    student = Student(name, events, results)

    # Test that Student instance is initialized correctly
    assert student.name == name
    assert student.events.equals(events)
    assert student.results.equals(results)
    assert student.nb_events == 2
    assert student.individual_average is None
    assert student.group_work_average is None
    assert student.overall_average is None
    assert student.grade is None


def test_student_repr(sample_student_data):

    # Get test data
    name, events, results = sample_student_data

    # Generate test student
    student = Student(name, events, results)

    # Expected string format
    expected_output = f"Student(name='{student.name}', nb_events=2, average=80, grade='B')"

    assert repr(student) == expected_output


def test_student_get_exam_results(sample_student_data):

    # Get test data
    name, events, results = sample_student_data

    # Generate test student
    student = Student(name, events, results)

    # Generate mock results DataFrame
    expected_output = pd.DataFrame({COL_NAME_EXAM + "1": [90],
                                    COL_NAME_EXAM + "2": [85],
                                    COL_NAME_FINAL: [95]})

    assert student.get_exam_results().equals(expected_output)


def test_student_get_outcome(sample_student_data):

    # Get test data
    name, events, results = sample_student_data

    # Generate test student
    student = Student(name, events, results)

    # Test case 1: grade is "E"
    student.grade = OUTCOME_GRADE_ECHEC
    assert student.get_outcome() == OUTCOME_ECHEC

    # Test case 2: grade is "XX"
    student.grade = OUTCOME_GRADES_ABANDON[0]
    assert student.get_outcome() == OUTCOME_ABANDON

    # Test case 3: grade is "AX"
    student.grade = OUTCOME_GRADES_ABANDON[1]
    assert student.get_outcome() == OUTCOME_ABANDON

    # Test case 4: grade is not "E", "XX", or "AX"
    student.grade = LETTER_GRADE_B
    assert student.get_outcome() == OUTCOME_SUCCES
