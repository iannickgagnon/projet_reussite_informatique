
# External libraries
import pandas as pd
from typing import Tuple
from os.path import isfile
from typing import Iterable

# Internal constants
from constants import (
    PATH_RESULTS,
    COL_NAME_NAME,
    COL_NAME_AVERAGE,
    COL_NAME_GRADE,
    _100_PERCENT,
    _0_PERCENT,
    RESULTS_TOKENS_GROUP_PROJECTS,
    RESULTS_TOKENS_MIDTERMS,
    RESULTS_TOKENS_FINAL_EXAM,
    RESULTS_TEAM_FLAG,
    RESULTS_IDX_WEIGHT,
    RESULTS_IDX_CORRECTED_ON,
    RESULTS_IDX_GROUP_PROJECT,
    RESULTS_COL_IDX_GRADE,
)


def __get_evaluation_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the evaluation structure from the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the evaluation structure.

    Returns:
        (pd.DataFrame): The extracted evaluation structure.
    """

    # Extract evaluation structure
    df_structure = df.iloc[6: 8, 3: df.shape[1] - 2]

    # Build header (list cast to remove name attribute which may or may not exist)
    df_structure.columns = list(df.iloc[5, 3: df.shape[1] - 2])

    # Remove unused columns
    df_structure = df_structure.drop(df_structure.columns[2:6], axis=1)

    # Convert to floating point
    df_structure = df_structure.astype(float)

    # Rename rows
    df_structure.index = [RESULTS_IDX_CORRECTED_ON, RESULTS_IDX_WEIGHT]

    # Validate evaluation structure
    sum_of_weights = df_structure.iloc[1, :].sum()
    assert sum_of_weights == 100, f'Sum of evaluation weights not equal to 100 ({sum_of_weights})'

    return df_structure


def any_token_in_string(string: str, tokens: Iterable[str]) -> bool:
    """
    Check if any of the given tokens is present in the given string.

    Args:
        string (str): The string to check.
        tokens (Iterable[str]): The tokens to look for.

    Returns:
        (bool): True if any of the tokens is present in the string, False otherwise.
    """

    return any(token in string for token in tokens)


def __standardize_structure_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the column names in the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to standardize.

    Returns:
        (pd.DataFrame): The DataFrame with standardized column names.
    """

    # Initialize counters
    tp_count = 1
    exam_count = 1

    # Eliminate spaces and convert to uppercase for column names
    df.columns = pd.Index([col.upper().replace(' ', '') for col in df.columns])

    for col_index, col_name in enumerate(df.columns):

        # Note: Finals need to be processed before midterms to avoid ambiguous names such as 'Examen Final'.

        # Build mapping
        if any_token_in_string(col_name, RESULTS_TOKENS_GROUP_PROJECTS):
            df.columns.values[col_index] = f'TP0{tp_count}'
            tp_count += 1
        elif any_token_in_string(col_name, RESULTS_TOKENS_FINAL_EXAM):
            df.columns.values[col_index] = 'FINAL'
        elif any_token_in_string(col_name, RESULTS_TOKENS_MIDTERMS):
            df.columns.values[col_index] = f'EXAM0{exam_count}'
            exam_count += 1

    return df


def __add_team_work_flag_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a row to the given DataFrame that contains a flag for group projects.

    Args:
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        (pd.DataFrame): The modified DataFrame.
    """

    df.loc[RESULTS_IDX_GROUP_PROJECT] = [True if 'TP' in col else False for col in df.columns]

    return df


def __get_grades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the grades from the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the grades.

    Returns:
        (pd.DataFrame): The extracted grades.
    """

    # Extract teams flag
    df_team_work_flag = df.iloc[8, 3::]

    # Extract grades
    df_grades = df.iloc[9:df.shape[0] - 7, 3::]

    # Remove unused columns but keep the last two for averages and letter grades
    cols_to_drop = [2, 3, 4, 5]
    for j in range(2, df_grades.shape[1] - 2):

        is_team_work_flag = df_team_work_flag.iat[j] == RESULTS_TEAM_FLAG
        is_nan = pd.to_numeric(df_grades.iloc[:, j], errors='coerce').isnull().all()

        if is_team_work_flag or is_nan:
            cols_to_drop.append(j)

    # Combine first two columns
    df_grades[df_grades.columns[0]] = df_grades[df_grades.columns[1]] + ' ' + df_grades[df_grades.columns[0]]

    # Mark second column for deletion
    cols_to_drop.append(1)

    # Drop marked columns
    df_grades.drop(df_grades.columns[cols_to_drop], axis=1, inplace=True)

    # Convert middle columns to floating point
    for col in df_grades.columns[1:-1]:
        df_grades[col] = df_grades[col].astype(float)

    # Remove trailing spaces in grades column
    df_grades[df_grades.columns[RESULTS_COL_IDX_GRADE]] = \
        df_grades[df_grades.columns[RESULTS_COL_IDX_GRADE]].apply(str.strip)

    # Reset indexes
    df_grades.reset_index(drop=True, inplace=True)

    return df_grades


def __remove_rows_of_nans_grades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows from the given DataFrame where the grades are just NaNs.

    Args:
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        (pd.DataFrame): The modified DataFrame.
    """

    rows_to_remove = []
    for i in range(df.shape[0]):
        if pd.isna(df[COL_NAME_AVERAGE].iat[i]):
            rows_to_remove.append(i)
    return df.drop(rows_to_remove)


def __standardize_grades_column_names(df: pd.DataFrame, structure_columns: pd.Index) -> pd.DataFrame:
    """
    Standardize the column names in the given grades DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to standardize.
        structure_columns (pd.Index): The index of the evaluation structure columns.

    Returns:
        (pd.DataFrame): The standardized grades DataFrame.
    """

    df.columns = pd.Index([COL_NAME_NAME] + list(structure_columns) + [COL_NAME_AVERAGE, COL_NAME_GRADE])

    return df


def __normalize_grades(df_structure: pd.DataFrame, df_grades: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the grades in the given DataFrame based on the evaluation structure.

    Args:
        df_structure (pd.DataFrame): The evaluation structure DataFrame.
        df_grades (pd.DataFrame): The grades DataFrame to normalize.

    Returns:
        (pd.DataFrame): The normalized grades DataFrame.
    """
    for col in df_structure.columns:
        if df_structure[col].loc[RESULTS_IDX_CORRECTED_ON] != _100_PERCENT and \
           df_structure[col].loc[RESULTS_IDX_CORRECTED_ON] != _0_PERCENT:
            df_grades[col] *= 100 / df_structure[col].loc[RESULTS_IDX_CORRECTED_ON]

            if any(df_grades[col] > 100):
                pass

    return df_grades


def __build_evaluation_structure_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the evaluation structure data from the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the evaluation structure.

    Returns:
        (pd.DataFrame): The evaluation structure DataFrame.
    """

    # Extract evaluation structure data
    df_structure = __get_evaluation_structure(df)

    # Fix column names
    df_structure = __standardize_structure_column_names(df_structure)

    # Add a row that contains a flag for group projects
    df_structure = __add_team_work_flag_row(df_structure)

    return df_structure


def __build_grades_data(df: pd.DataFrame, df_structure: pd.DataFrame) -> pd.DataFrame:
    """
    Build the grades data from the given DataFrames.

    Args:
        df (pd.DataFrame): The DataFrame containing the grades.
        df_structure (pd.DataFrame): The DataFrame containing the evaluation structure.

    Returns:
        (pd.DataFrame): The grades DataFrame.
    """

    # Extract grades
    df_grades = __get_grades(df)

    # Fix column names
    df_grades = __standardize_grades_column_names(df_grades, df_structure.columns)

    # Remove rows where grades are just NaNs
    df_grades = __remove_rows_of_nans_grades(df_grades)

    # Adjust grades
    df_grades = __normalize_grades(df_structure, df_grades)

    # Replace NaNs with zeros
    df_grades.fillna(0.0, inplace=True)

    return df_grades


def parse_results(filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse the given results file.

    Args:
        filename (str): The name of the results file.

    Returns:
        (Tuple[pd.DataFrame, pd.DataFrame]): A tuple containing the evaluation structure DataFrame and the grades.

    Raises:
        AssertionError: If the sum of the evaluation weights is not equal to 100.
    """

    # Adjust filename
    if not isfile(filename):
        filename = PATH_RESULTS + filename
        if not isfile(filename):
            raise FileExistsError('Results file not found')

    # Read raw file
    df_raw = pd.read_csv(filename, sep=';')

    # Extract evaluation structure
    df_structure = __build_evaluation_structure_data(df_raw)

    # Extract grades
    df_grades = __build_grades_data(df_raw, df_structure)

    return df_structure, df_grades
