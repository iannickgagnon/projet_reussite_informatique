# External libraries
from unittest.mock import patch
import pandas as pd
import pytest

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
    COL_NAME_CONTEXT,
    COL_NAME_NAME,
    COL_NAME_TIME,
    COL_NAME_DATE,
)


@pytest.fixture
def test_df_context():
    return pd.DataFrame({COL_NAME_CONTEXT: ['INF136-01 A2022', '', '', ''],
                         COL_NAME_NAME: ['Amel', 'Éric', 'Amel', 'Iannick']})


def test_get_course_id(test_df_context):

    # Checks that the method fetches the right course and converts dashes with underscores.

    assert __get_course_id(test_df_context) == 'INF136_01'


def test_get_course_id_no_match(test_df_context):

    # Test data
    test_df_context = pd.DataFrame({COL_NAME_CONTEXT: ['Not a match']})

    with pytest.raises(ValueError):
        __get_course_id(test_df_context)


def test_get_semester(test_df_context):
    assert __get_semester(test_df_context) == 'A2022'


def test_get_semester_no_match(test_df_context):

    # Remove semester id from fixture
    test_df_context.loc[0, COL_NAME_CONTEXT] = 'This is a context without a semester id'

    with pytest.raises(ValueError):
        __get_semester(test_df_context)


def test_time_str_to_float():
    assert __time_str_to_float('00:30') == 0.5
    assert __time_str_to_float('01:15') == 1.25
    assert __time_str_to_float('02:45') == 2.75


def test_get_members_list(test_df_context):

    # Expected unique names
    expected_names = ['Amel', 'Éric', 'Iannick']

    # Extract unique member names
    obtained_names = __get_members_list(test_df_context)

    assert set(obtained_names) == set(expected_names)


def test_get_members_list_no_match():

    # Create dummy context without a name column
    df_no_match = pd.DataFrame({'No match': []})

    with pytest.raises(KeyError):
        __get_members_list(df_no_match)


def test_parse_events():

    # Mock DataFrame
    mock_data = pd.DataFrame({
        COL_NAME_TIME: ['2022-01-01, 09:00',
                        '2022-01-01, 10:30',
                        '2022-01-01, 11:00',
                        '2022-01-02, 12:00'],
        COL_NAME_NAME: ['Amel', 'Eric', 'Iannick', 'Jacob'],
        COL_NAME_CONTEXT: ['INF136-01 A2022'] * 4
    })

    # Mocking expected data for pd.read_csv
    expected_columns = [COL_NAME_TIME, COL_NAME_NAME, COL_NAME_DATE]
    expected_data = [
        [9.0, 'Amel', '2022-01-01'],
        [10.5, 'Eric', '2022-01-01'],
        [11.0, 'Iannick', '2022-01-01'],
        [12.0, 'Jacob', '2022-01-02']
    ]

    expected_df = pd.DataFrame(expected_data, columns=expected_columns)

    # Expected outputs
    expected_course_id = 'INF136_01'
    expected_semester = 'A2022'

    # Parse while mocking pd.read_csv
    with patch('pandas.read_csv', return_value=mock_data):
        actual_data, actual_course_id, actual_semester = parse_events('not_a_filename.csv')

    # Validate course ID and semester
    assert actual_course_id == expected_course_id
    assert actual_semester == expected_semester

    # Validate column names in a manner insensitive to order
    assert actual_data.columns.equals(pd.Index(expected_columns))

    # Validate content
    assert actual_data.equals(expected_df)
