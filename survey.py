
# External libraries
from copy import deepcopy
from os.path import isfile

# Internal libraries
from survey_answers import SurveyAnswers

# Internal constants
from constants import (
    PATH_SURVEYS,
    SURVEY_NB_QUESTIONS,
    SURVEY_LINE_LENGTH_COMPLETE,
    SURVEY_LINE_ANSWER_Q1,
    SURVEY_LINE_FIRST_NAME,
    SURVEY_LINE_LAST_NAME,
    SURVEY_NO_QUESTION_NB_WORK_HOURS,
)


class Survey:
    """
    A class representing a survey.

    Attributes:
        answer_sets (list): A list of SurveyAnswers objects representing the answer sets for the survey.
        count (int): An integer representing the number of survey instances.
    """

    def __init__(self, filename: str = None, answers: list = None):
        """
        Initializes a SurveyAnswers object.

        If answers are not provided, reads data from a CSV file and delegates complex build process to __build method.

        Args:
            filename (str): The name of the CSV file containing the survey data.
            answers (list[SurveyAnswers]): A list of SurveyAnswers objects representing the survey data.
        """

        if answers is None:

            # Initialize storage for answers
            self.answer_sets = []

            # Delegate complex build process
            self.__build(filename)

        else:

            self.answer_sets = answers

        # Store number of survey instances
        self.count = len(self.answer_sets)

    def __build(self, filename):
        """
       Reads data from a CSV file, parses the lines, and stores the cleaned up answers in a list of SurveyAnswers.

       Args:
           filename (str): The name of the CSV file containing the survey data.
       """

        # Adjust filename
        if not isfile(filename):
            filename = PATH_SURVEYS + filename

        # Read data
        with open(filename, encoding='utf-8') as file:
            text = file.readlines()

        # Parse lines except header
        for ligne in text[1:]:

            # Split current line
            parts = ligne.split(',')

            # Ignore if incomplete
            if len(parts) != SURVEY_LINE_LENGTH_COMPLETE:
                continue

            # Store cleaned up answers
            indexed_answers = enumerate(parts[SURVEY_LINE_ANSWER_Q1: SURVEY_LINE_ANSWER_Q1 + SURVEY_NB_QUESTIONS])
            current_answers = {i: response[1:len(response) - 1].replace('"', '') for i, response in indexed_answers}

            # Build current student name
            current_student_name = \
                f'{parts[SURVEY_LINE_FIRST_NAME]} {parts[SURVEY_LINE_LAST_NAME]}'.replace('"', '').replace('\n', '')

            # Add filled-in answers to list
            if all(current_answers.values()):

                # Convert number of work hours to integer
                if current_answers[SURVEY_NO_QUESTION_NB_WORK_HOURS].isdigit():
                    current_answers[SURVEY_NO_QUESTION_NB_WORK_HOURS] = \
                        int(current_answers[SURVEY_NO_QUESTION_NB_WORK_HOURS])

                # Build current survey answers object
                current_survey_answers = SurveyAnswers(current_answers, current_student_name)

                # Add to list of answers
                self.answer_sets.append(current_survey_answers)

    def __getitem__(self, index: int):
        """
        Returns the SurveyAnswers object at the given index.

        Args:
            index (int): The index of the SurveyAnswers object to return.

        Returns:
            SurveyAnswers: The SurveyAnswers object at the given index.
        """

        try:
            return self.answer_sets[index]
        except IndexError:
            print('Invalid answer set index')

    def __delitem__(self, index: int):
        """
        Deletes the SurveyAnswers object at the given index and decrements the count of answer sets.

        Args:
            index (int): The index of the SurveyAnswers object to delete.
        """

        try:

            # Remove desired answer set
            del self.answer_sets[index]

            # Decrement answer sets count
            self.count -= 1

        except IndexError:
            print('Invalid answer set index')

    def __iter__(self):
        """
        Initializes an iterator position.

        Returns:
            SurveyAnswers: The next SurveyAnswers object in the list.
        """

        # Initialize iterator position
        self.__index = 0
        return self

    def __next__(self):
        """
        Returns the next SurveyAnswers object in the list.

        Returns:
            SurveyAnswers: The next SurveyAnswers object in the list.
        """

        if self.__index >= self.count:

            # Expected for end of loop
            raise StopIteration

        else:

            # Extract next answer
            answer = self.answer_sets[self.__index]

            # Increment iterator
            self.__index += 1

            return answer

    def __len__(self):
        """
        Returns the number of SurveyAnswers objects in the answer_sets list.

        Returns:
            int: The number of SurveyAnswers objects in the answer_sets list.
        """
        return len(self.answer_sets)

    def __repr__(self):
        """
        Returns a string representation of the Survey object.

        Returns:
            str: The string representation of the Survey object.
        """

        # Extract keys and values
        keys = tuple(self[0].answers.keys())
        values = tuple(self[0].answers.values())

        # Build answers dictionary string representation
        answers_str = '{' + f'{keys[0]}:{values[0]}, ..., {keys[-1]}:{values[-1]}, ...' + '}'

        return f'Survey(count: {self.count}, answers: {answers_str} )'

    def filter_by_student_name(self, student_name: str):
        """
           Returns a new Survey object containing SurveyAnswers objects for a given student name.

           Args:
               student_name (str): The name of the student to filter by.

           Returns:
               Survey: The Survey of the given student.
        """

        # Extract answers for given student
        answers = [answers for answers in self if answers.student_name == student_name]

        # Signal invalid student name or build new survey
        if len(answers) == 0:
            return None
        else:
            return Survey(answers=[answers for answers in self if answers.student_name == student_name])


    @staticmethod
    def build_compiled_survey_results_data_structure() -> dict:

        # Build outcome data structure
        OUTCOMES = {outcome: 0 for outcome in ('Abandon', 'Succès', 'Échec')}

        # Build yes/no question data structure
        YES_NO = {'Oui': OUTCOMES.copy(),
                  'Non': OUTCOMES.copy()}

        # Build summer job data structure
        EMPLOI_ETE = {'Études seulement': OUTCOMES.copy(),
                      'Travail seulement': OUTCOMES.copy(),
                      'Études et travail': OUTCOMES.copy()}

        # Build language data structure
        LANGUE = {'Français': OUTCOMES.copy(),
                  'Anglais': OUTCOMES.copy(),
                  'Autre': OUTCOMES.copy()}

        # Build financial situation data structure
        SITUATION_FINANCIERE = {'Aisée': OUTCOMES.copy(),
                                'Satisfaisante': OUTCOMES.copy(),
                                'Précaire': OUTCOMES.copy()}

        # Build compiled survey results data structure
        questions_and_outcomes = {0: deepcopy(YES_NO),
                                  1: deepcopy(YES_NO),
                                  2: deepcopy(YES_NO),
                                  3: deepcopy(YES_NO),
                                  4: deepcopy(YES_NO),
                                  5: deepcopy(EMPLOI_ETE),
                                  6: [],
                                  7: deepcopy(LANGUE),
                                  8: deepcopy(LANGUE),
                                  9: deepcopy(YES_NO),
                                  10: deepcopy(YES_NO),
                                  11: deepcopy(SITUATION_FINANCIERE)}

        return questions_and_outcomes
