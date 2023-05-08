
# External libraries
import pytest
import pandas as pd
from unittest.mock import patch

# Internal libraries
from results_parser import (
    __build_evaluation_structure_data,
    __build_grades_data,
    parse_results,
)

# Internal constants
from constants import (
    PATH_TEST_RESULTS_FROM_TESTS,
)


@pytest.fixture
def test_data_raw():

    df = pd.read_csv(PATH_TEST_RESULTS_FROM_TESTS, sep=';')
    return df


@pytest.fixture
def test_data_expected_structure():

    df = pd.DataFrame({
        "EXAM01": [8.5, 10.0, 0.0],
        "EXAM02": [100, 10.0, 0.0],
        "TP01": [100.0, 10.0, 1.0],
        "FINAL": [100.0, 40.0, 0.0],
        "TP02": [100.0, 15.0, 1.0],
        "RAP01": [8.0, 0.0, 0.0],
        "TP03": [100.0, 15.0, 1.0]
    }, index=["Corrected on", "Weight", "Group project"])

    return df


@pytest.fixture
def test_data_expected_grades():

    df = pd.DataFrame({
        "Name": ["John Doe", "Jane Doe", "Invalid Student"],
        "EXAM01": [100.0, 50.0, 0.0],
        "EXAM02": [100.0, 50.0, 0.0],
        "TP01": [100.0, 50.0, 0.0],
        "FINAL": [100.0, 50.0, 0.0],
        "TP02": [100.0, 50.0, 0.0],
        "RAP01": [100.0, 50.0, 0.0],
        "TP03": [100.0, 50.0, 0.0],
        "Average": [100.0, 50.0, 0.0],
        "Grade": ["A+", "C", "E"],
    })

    return df


def test_build_evaluation_structure_data(test_data_raw: pd.DataFrame, test_data_expected_structure: pd.DataFrame):

    # Get raw data DataFrame
    df = test_data_raw

    # Mock expected results DataFrame
    expected = test_data_expected_structure

    # Build evaluation structure from raw data DataFrame
    result = __build_evaluation_structure_data(df)

    assert result.equals(expected)


def test_build_grades_data(test_data_raw: pd.DataFrame, test_data_expected_grades: pd.DataFrame):

    # Get raw data DataFrame
    df = test_data_raw

    # Mock expected results DataFrame
    expected = test_data_expected_grades

    # Build evaluation structure from raw data DataFrame (tested separately in test_build_evaluation_structure_data)
    df_structure = __build_evaluation_structure_data(df)

    # Build grades data from raw data and evaluation structure DataFrames
    result = __build_grades_data(df, df_structure)

    assert result.equals(expected)


def test_parse_results(test_data_raw: pd.DataFrame,
                       test_data_expected_structure: pd.DataFrame,
                       test_data_expected_grades: pd.DataFrame):

    with patch('pandas.read_csv', return_value=test_data_raw):
        obtained_structure, obtained_grades = parse_results('not_a_filename.csv')

    assert obtained_structure.equals(test_data_expected_structure)
    assert obtained_grades.equals(test_data_expected_grades)
