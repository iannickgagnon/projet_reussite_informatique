import pandas as pd
from typing import Iterable
from typing import Tuple

_TOKENS_GROUP_PROJECTS = ('TP', 'TRAVAIL', 'PRATIQUE', 'DEV')
_TOKENS_MIDTERMS = ('INTRA', 'EXAM')
_TOKENS_FINAL_EXAM = ('FINAL',)

_INDEX_CORRECTED_ON = 'Corrected on'
_INDEX_WEIGHT = 'Weight'
_INDEX_GROUP_PROJECT = 'Group project'

_COL_INDEX_GRADE = -1

_COL_NAME_NAME = 'Name'
_COL_NAME_AVERAGE = 'Average'
_COL_NAME_GRADE = 'Grade'

_100_PERCENT = 100
_0_PERCENT = 0

_TEAM_FLAG = 'Équ.'

def _get_evaluation_structure(df: pd.DataFrame) -> pd.DataFrame:
    # Extract evaluation structure
    df_structure = df.iloc[6: 8, 3: df.shape[1] - 2]

    # Build header (list cast to remove name attribute which may or may not exist)
    df_structure.columns = list(df.iloc[5, 3: df.shape[1] - 2])

    # Remove unused columns
    df_structure = df_structure.drop(df_structure.columns[2:6], axis=1)

    # Convert to floating point
    df_structure = df_structure.astype(float)

    # Rename rows
    df_structure.index = [_INDEX_CORRECTED_ON, _INDEX_WEIGHT]

    # Validate evaluation structure
    sum_of_weights = df_structure.iloc[1, :].sum()
    assert sum_of_weights == 100, f'Sum of evaluation weights not equal to 100 ({sum_of_weights})'

    return df_structure


def _any_token_in_string(string: str, tokens: Iterable[str]) -> bool:
    return any([token in string for token in tokens])


def _standardize_structure_column_names(df: pd.DataFrame) -> pd.DataFrame:
    # Initialize counters
    tp_count = 1
    exam_count = 1

    # Eliminate spaces and convert to uppercase for column names
    df.columns = [col.upper().replace(' ', '') for col in df.columns]

    # Initialize mapping for column names
    mapping = dict()

    for col in df.columns:

        # FIXME: Finals need to be processed before midterms to avoid ambiguous names such as 'Examen Final'.

        # Build mapping
        if _any_token_in_string(col, _TOKENS_GROUP_PROJECTS):
            mapping[col] = f'TP0{tp_count}'
            tp_count += 1
        elif _any_token_in_string(col, _TOKENS_FINAL_EXAM):
            mapping[col] = 'FINAL'
        elif _any_token_in_string(col, _TOKENS_MIDTERMS):
            mapping[col] = f'EXAM0{exam_count}'
            exam_count += 1

    # Apply mapping
    df.rename(columns=mapping, inplace=True)

    return df


def _add_team_work_flag_row(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[_INDEX_GROUP_PROJECT] = [True if 'TP' in col else False for col in df.columns]

    return df


def _get_grades(df: pd.DataFrame, df_structure: pd.DataFrame) -> pd.DataFrame:

    # Extract teams flag
    df_team_work_flag = df.iloc[8, 3::]

    # Extract grades
    df_grades = df.iloc[9:df.shape[0] - 7, 3::]

    # Remove unused columns but keep the last two for averages and letter grades
    cols_to_drop = [2, 3, 4, 5]
    for j in range(2, df_grades.shape[1] - 2):

        is_team_work_flag = df_team_work_flag.iat[j] == _TEAM_FLAG
        is_nan = pd.isna(df_grades.iat[0, j])

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
    df_grades[df_grades.columns[_COL_INDEX_GRADE]] = df_grades[df_grades.columns[_COL_INDEX_GRADE]].apply(str.strip)

    # Reset indexes
    df_grades.reset_index(drop=True, inplace=True)

    return df_grades


def _remove_rows_of_nans_grades(df: pd.DataFrame) -> pd.DataFrame:
    rows_to_remove = []
    for i in range(df.shape[0]):
        if pd.isna(df[_COL_NAME_AVERAGE].iat[i]):
            rows_to_remove.append(i)
    return df.drop(rows_to_remove)


def _standardize_grades_column_names(df: pd.DataFrame, structure_columns: pd.Index) -> pd.DataFrame:
    df.columns = pd.Index([_COL_NAME_NAME] + list(structure_columns) + [_COL_NAME_AVERAGE, _COL_NAME_GRADE])
    return df


def _normalize_grades(df_structure: pd.DataFrame, df_grades: pd.DataFrame) -> pd.DataFrame:
    for col in df_structure.columns:
        if df_structure[col].loc[_INDEX_CORRECTED_ON] != _100_PERCENT and \
           df_structure[col].loc[_INDEX_CORRECTED_ON] != _0_PERCENT:
            df_grades[col] *= 100 / df_structure[col].loc[_INDEX_CORRECTED_ON]

        if any(df_grades[col] > 100):
            pass

    return df_grades


def _build_evaluation_structure_data(df: pd.DataFrame) -> pd.DataFrame:

    # Extract evaluation structure data
    df_structure = _get_evaluation_structure(df)

    # Fix column names
    df_structure = _standardize_structure_column_names(df_structure)

    # Add a row that contains a flag for group projects
    df_structure = _add_team_work_flag_row(df_structure)

    return df_structure


def _build_grades_data(df: pd.DataFrame, df_structure: pd.DataFrame) -> pd.DataFrame:
    # Extract grades
    df_grades = _get_grades(df, df_structure)

    # Fix column names
    df_grades = _standardize_grades_column_names(df_grades, df_structure.columns)

    # Remove rows where grades are just NaNs
    df_grades = _remove_rows_of_nans_grades(df_grades)

    # Adjust grades
    df_grades = _normalize_grades(df_structure, df_grades)

    return df_grades


def parse_results(filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Read raw file
    df_raw = pd.read_csv(filename, sep=';')

    # Extract evaluation structure
    df_structure = _build_evaluation_structure_data(df_raw)

    # Extract grades
    df_grades = _build_grades_data(df_raw, df_structure)

    # Replace NaNs with zeros
    df_grades.fillna(0.0, inplace=True)

    return df_structure, df_grades
