# External libraries
import pandas as pd

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
)


class Student:
    """
    A class representing a student.

    Attributes:

        name (str): The name of the student.
        events (DataFrame): A DataFrame containing information about the student's events.
        results (DataFrame): A DataFrame containing the student's results.
        nb_events (int): The number of events attended by the student.
        individual_average (float): The student's individual average.
        group_work_average (float): The student's group work average.
        overall_average (float): The student's overall average.
        grade (str): The student's grade.

    Methods:

        __init__(self, name: str, events: pd.DataFrame, results: pd.DataFrame) -> None:
            Initializes a new instance of the Student class.

        __repr__(self) -> str:
            Returns a string representation of the Student instance.

        get_exam_results(self) -> pd.DataFrame:
            Returns a DataFrame containing the student's exam results.

        get_outcome(self) -> str:
            Returns the outcome of the student.
    """

    def __init__(self,
                 name: str,
                 events: pd.DataFrame,
                 results: pd.DataFrame) -> None:
        """
        Initializes a new instance of the Student class.

        Args:
            name (str): The name of the student.
            events (DataFrame): A DataFrame containing information about the student's events.
            results (DataFrame): A DataFrame containing the student's results.
        """

        # Store raw data
        self.name = name
        self.events = events
        self.results = results

        # Store dimensions
        self.nb_events = events.shape[0]

        # Initialize attributes to be calculated
        self.individual_average = None
        self.group_work_average = None
        self.overall_average = None
        self.grade = None

    def __repr__(self):
        """
        Returns a string representation of the Student instance.

        Returns:
            str: A string representation of the Student instance.
        """

        # Extract student's information
        name = self.name
        nb_events = self.nb_events
        average = self.results[COL_NAME_AVERAGE].iloc[0]
        grade = self.results[COL_NAME_GRADE].iloc[0]

        return f"Student(name='{name}', nb_events={nb_events}, average={average}, grade='{grade}')"

    def get_exam_results(self) -> pd.DataFrame:
        """
          Returns a DataFrame containing the student's exam results.

          Returns:
              DataFrame: A DataFrame containing the student's exam results.
          """

        # Extract midterms and final exam grades
        exams = self.results.filter(like=COL_NAME_EXAM)
        final = self.results.filter(like=COL_NAME_FINAL)

        return pd.concat([exams, final], axis=1)

    def get_outcome(self) -> str:
        """
         Returns the outcome of the student such as 'Échec', 'Abandon' and 'Succès' based on his or her letter grade.

         Returns:
             str: The outcome of the student.
         """

        if self.grade == OUTCOME_GRADE_ECHEC:
            return OUTCOME_ECHEC
        elif self.grade in OUTCOME_GRADES_ABANDON:
            return OUTCOME_ABANDON
        else:
            return OUTCOME_SUCCES
