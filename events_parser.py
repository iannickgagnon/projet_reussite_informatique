"""
Parses an event file and returns relevant information.

Args:
    filename (str): The name of the file to be parsed.

Returns:
    tuple[DataFrame, str, str]: Tuple containing the parsed DataFrame, course ID , and the semester identifier.

Raises:
    FileNotFoundError: If the specified file cannot be found.
    ValueError: If the specified file is not in the expected format.

External libraries:
    - pandas
    - re

Internal constants:
    - COLS_TO_REMOVE (list): A list of columns to be removed from the parsed DataFrame.
    - COLS_TO_RENAME (dict): A dictionary mapping original column names to new column names.
    - ORIGINAL_COL_NAME_NAME (str): The original name of the column containing member names.
    - ORIGINAL_COL_NAME_TIME (str): The original name of the column containing time stamps.
    - COL_NAME_CONTEXT (str): The name of the column containing context information.
    - COL_NAME_DATE (str): The name of the new column containing dates.
    - COL_NAME_TIME (str): The name of the new column containing floating-point time values.

Internal functions:
    - __get_course_id(data: DataFrame) -> str: Extracts course ID from the context information in the parsed DataFrame.
    - __get_semester(data: DataFrame) -> str: Extracts semester from the context information in the parsed DataFrame.
    - __time_str_to_float(time: str) -> float: Converts a time string in the format HH:MM to a floating-point number.
    - __get_members_list(data: DataFrame) -> list: Returns a list of unique member names in the parsed DataFrame.
"""

# External libraries
import pandas as pd
import re

# Internal constants
from constants import (
    COLS_TO_REMOVE,
    COLS_TO_RENAME,
    ORIGINAL_COL_NAME_NAME,
    ORIGINAL_COL_NAME_TIME,
    COL_NAME_CONTEXT,
    COL_NAME_DATE,
    COL_NAME_TIME,
    COL_NAME_NAME,
)


def __get_course_id(data: pd.DataFrame) -> str:
    """
    Extracts the course ID from the context string in the parsed DataFrame.

    Args:
        data (DataFrame): The parsed DataFrame containing the context information.

    Returns:
        str: The course ID in string format.

    Raises:
        ValueError: If the course ID pattern cannot be found in the context string.
    """

    # Get context string
    context_str = data.loc[0].at[COL_NAME_CONTEXT]

    # Extract course ID as a re.Match object
    course_id_match = re.search('[A-Z]{3}[0-9]{3}-[0-9]{2}', context_str)

    if course_id_match is None:
        raise ValueError('Could not find course ID in context string')
    else:
        return course_id_match.group().replace('-', '_')


def __get_semester(data: pd.DataFrame) -> str:
    """
    Extracts the semester from the given DataFrame.

    Args:
        data (DataFrame): A DataFrame containing context information.

    Returns:
        str: A string representing the semester.

    Raises:
        AttributeError: If no match is found for the regular expression pattern.
    """

    # Get context string
    string = data.loc[0].at[COL_NAME_CONTEXT]

    # Extract semester identifier
    semester = re.search('[A-Z][0-9]{4}', string)

    if semester is None:
        raise ValueError('Could not find semester id in context string')
    else:
        return semester.group()


def __time_str_to_float(time: str) -> float:
    """
    Converts a time string in 'HH:MM' format to a floating point number.

    Args:
        time (str): A string containing the time in 'HH:MM' format.

    Returns:
        float: A floating point number representing the time in hours, with fractional parts for minutes.
    """

    # Extract hours and minutes
    h, m = time.split(':')

    # Convert to float
    time_float = float(h) + float(m) / 60.

    return time_float


def __get_members_list(data: pd.DataFrame) -> list:
    """
    Extracts a list of unique member names from a DataFrame.

    Args:
      data (DataFrame): A DataFrame containing member data.

    Returns:
      list: A list of unique member names.
    """

    if ORIGINAL_COL_NAME_NAME in data.columns:
        return list(data[ORIGINAL_COL_NAME_NAME].unique())
    elif COL_NAME_NAME in data.columns:
        return list(data[COL_NAME_NAME].unique())
    else:
        raise KeyError('Could not find the name column')


def parse_events(filename: str) -> tuple[pd.DataFrame, str, str]:
    """
    Parses a CSV file containing event data and returns a cleaned DataFrame with course ID and semester information.

    Args:
        filename (str): The path to the CSV file containing event data.

    Returns:
        tuple: A tuple containing:
            - df (DataFrame): A cleaned DataFrame with columns for date, time, context, and member names.
            - course_id (str): The course ID extracted from the context string in the first row of the DataFrame.
            - semester (str): The semester extracted from the context string in the first row of the DataFrame.
    """

    # Import raw file
    df = pd.read_csv(filename)

    # Get course info before removing columns
    course_id = __get_course_id(df)
    semester = __get_semester(df)

    # Rename columns
    df = df.rename(columns=COLS_TO_RENAME)

    # Separate time stamps into separate dates and times
    if ORIGINAL_COL_NAME_TIME in df.columns:
        df[[COL_NAME_DATE, COL_NAME_TIME]] = \
            df[ORIGINAL_COL_NAME_TIME].apply(lambda x: pd.Series(str(x).split(',')))
    else:
        df[[COL_NAME_DATE, COL_NAME_TIME]] = \
            df[COL_NAME_TIME].apply(lambda x: pd.Series(str(x).split(',')))

    # Convert time stamps to floating point numbers
    df[COL_NAME_TIME] = \
        df[COL_NAME_TIME].apply(lambda x: __time_str_to_float(x))

    # Eliminate unused columns if present
    common_columns = list(set(COLS_TO_REMOVE).intersection(df.columns))
    df.drop(columns=common_columns, inplace=True)

    return df, course_id, semester
