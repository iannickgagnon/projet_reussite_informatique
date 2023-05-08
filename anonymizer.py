
# External libraries
from typing import Tuple
import pandas as pd

# Internal libraries
import survey

# Internal constants
from constants import (
    COL_NAME_NAME,
)


def anonymize(course_id: str,
              semester: str,
              events: pd.DataFrame = None,
              results: pd.DataFrame = None,
              surveys: survey.Survey = None,
              clean_up: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, survey.Survey]:
    """
    Anonymize student data for a given course and semester.

    Args:
        course_id (str): ID of the course to anonymize.
        semester (str): Semester of the course to anonymize.
        events (DataFrame, optional): Dataframe of events to anonymize. Defaults to None.
        results (DataFrame, optional): Dataframe of results to anonymize. Defaults to None.
        surveys (Survey, optional): Survey data to anonymize. Defaults to None.
        clean_up (bool, optional): If True, removes entries that could not be anonymized. Defaults to False.

    Returns:
        Tuple[DataFrame, DataFrame, Survey]: Tuple containing anonymous events, results, and Survey.

    Raises:
        TypeError: If the course_id or semester argument is not a string.
        ValueError: If the events, results, or surveys argument is None.
    """

    # Get list of members
    members = list(results[COL_NAME_NAME])
    nb_members = len(members)

    # Get prefix for anonymous name
    prefix = f'{course_id}_{semester}'

    # Build mapping
    names_dict = {original: f'{prefix}_{i}' for original, i in zip(members, range(nb_members))}

    # Substitute names for anonymous versions
    events[COL_NAME_NAME].replace(to_replace=names_dict, inplace=True)
    results[COL_NAME_NAME].replace(to_replace=names_dict, inplace=True)

    for name in names_dict:
        corresponding_survey = surveys.filter_by_student_name(name)
        if corresponding_survey is not None:
            corresponding_survey[0].student_name = names_dict[name]

    # Remove entries that could not be anonymized
    if clean_up:
        events, results, surveys = __clean_up(prefix, events, results, surveys)

    return events, results, surveys


def __clean_up(prefix, events=None, results=None, surveys=None):
    """
    Removes entries that could not be anonymized.

    Args:
        prefix (str): Prefix used for anonymous names.
        events (DataFrame, optional): DataFrame of events to clean up. Defaults to None.
        results (DataFrame, optional): DataFrame of results to clean up. Defaults to None.
        surveys (Survey, optional): Survey data to clean up. Defaults to None.

    Returns:
        Tuple[DataFrame, DataFrame, Survey]: Tuple containing the cleaned events, results, and Survey.
    """

    # Filter surveys
    surveys = [s for s in surveys if prefix in s.student_name]

    # Build list of names of students who have completed the survey
    student_names_to_keep = [answers.student_name for answers in surveys]

    # Filter events
    events = events[events[COL_NAME_NAME].isin(student_names_to_keep)]

    # Filter results
    results = results[results[COL_NAME_NAME].isin(student_names_to_keep)]

    return events, results, surveys
