
# External libraries
import pytest
import pandas as pd

# Internal libraries
from events_parser import (
    __get_course_id,
    __get_semester,
    __time_str_to_float,
    __get_members_list,
    parse_events,
)

# Internal constants
from constants import (
    PATH_TEST_EVENTS_FROM_TESTS,
    COL_NAME_CONTEXT,
    COL_NAME_NAME,
    COL_NAME_TIME,
    COL_NAME_DATE,
    COL_NAME_EVENT,
)


@pytest.fixture
def test_data_raw():

    df = pd.read_csv(PATH_TEST_EVENTS_FROM_TESTS)
    return df


@pytest.fixture
def test_data_expected_events():

    expected_columns = [COL_NAME_NAME, COL_NAME_EVENT, COL_NAME_DATE, COL_NAME_TIME]
    expected_data = [
        ['John Doe', 'Journal consulté', '2022-01-1', 9.0],
        ['Jane Doe', 'Cours consulté', '2022-01-1', 10.5],
        ['Invalid Student', 'Cours consulté', '2022-01-1', 17.5],
    ]

    df = pd.DataFrame(expected_data, columns=expected_columns)

    return df


def test_events_get_course_id(test_data_raw):

    course_id = __get_course_id(test_data_raw)

    assert course_id == 'INF135_02'


def test_get_course_id(test_data_raw):

    # Checks that the method fetches the right course and converts dashes with underscores.

    assert __get_course_id(test_data_raw) == 'INF135_02'


def test_get_course_id_no_match(test_data_raw):

    # Test data
    test_df_context = pd.DataFrame({COL_NAME_CONTEXT: ['Not a match']})

    with pytest.raises(ValueError):
        __get_course_id(test_df_context)


def test_get_semester(test_data_raw):
    assert __get_semester(test_data_raw) == 'A2022'


def test_get_semester_no_match(test_data_raw):

    # Remove semester id from fixture
    test_data_raw.loc[0, COL_NAME_CONTEXT] = 'This is a context without a semester id'

    with pytest.raises(ValueError):
        __get_semester(test_data_raw)


def test_time_str_to_float():
    assert __time_str_to_float('00:30') == 0.5
    assert __time_str_to_float('01:15') == 1.25
    assert __time_str_to_float('02:45') == 2.75


def test_get_members_list(test_data_raw):

    # Expected unique names
    expected_names = ['John Doe', 'Jane Doe', 'Invalid Student']

    # Extract unique member names
    obtained_names = __get_members_list(test_data_raw)

    assert set(obtained_names) == set(expected_names)


def test_get_members_list_no_match():

    # Create dummy context without a name column
    df_no_match = pd.DataFrame({'No match': []})

    with pytest.raises(KeyError):
        __get_members_list(df_no_match)


def test_parse_events(test_data_expected_events):

    # Expected outputs
    expected_course_id = 'INF135_02'
    expected_semester = 'A2022'

    # Parse data (parse_events is tested separately)
    actual_data, actual_course_id, actual_semester = parse_events(PATH_TEST_EVENTS_FROM_TESTS)

    # Validate course ID and semester
    assert actual_course_id == expected_course_id
    assert actual_semester == expected_semester

    # Validate column names in a manner insensitive to order
    assert actual_data.columns.equals(test_data_expected_events.columns)

    # Validate content
    assert actual_data.equals(test_data_expected_events)
