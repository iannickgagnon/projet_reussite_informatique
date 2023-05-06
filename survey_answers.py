
class SurveyAnswers:
    """
    A class representing the survey answers of a student.

    Attributes:
        answers (dict): A dictionary containing the survey question-answer pairs.
        nb_questions (int): The number of questions in the survey.
        student_name (str): The name of the student who provided the survey answers.

    Methods:

        init(self, answers: dict, student_name: str) -> None:
            Initializes a SurveyAnswers object.

        iter(self) -> SurveyAnswers:
            Initializes an iterator position.

        next(self) -> dict:
            Returns the next survey answer in the iteration.

        getitem(self, index: int) -> dict:
            Returns the survey answer at the specified index.

        repr(self) -> str:
            Returns the string representation of the SurveyAnswers object.
    """

    def __init__(self, answers: dict, student_name: str):
        """
        Initializes a SurveyAnswers object.

        Args:
            answers (dict): A dictionary containing the survey question-answer pairs.
            student_name (str): The name of the student who provided the survey answers.
        """

        self.answers = answers
        self.nb_questions = len(answers)
        self.student_name = student_name

    def __iter__(self):
        """
          Initializes an iterator position.
        """
        self.__index = 0
        return self

    def __next__(self):
        """
        Returns the next survey answer in the iteration.

        Raises:
            StopIteration: If there are no more survey answers to iterate through.
        """

        if self.__index >= self.nb_questions:

            # Expected for end of loop
            raise StopIteration

        else:

            # Extract next answer
            answer = self.answers[self.__index]

            # Increment iterator
            self.__index += 1

            return answer

    def __getitem__(self, index: int):
        """
        Returns the survey answer at the specified index.

        Args:
            index (int): The index of the survey answer to retrieve.

        Returns:
            (dict): The survey answer at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """

        # Since the data is stored in a dictionary, catch KeyError and "convert" to IndexError
        try:
            return self.answers[index]
        except KeyError:
            raise IndexError('Invalid answer index')

    def __repr__(self):
        """
        Returns the string representation of the SurveyAnswers object.

        Returns:
            (str): The string representation of the SurveyAnswers object.
        """

        # Extract keys and values
        keys = tuple(self.answers.keys())
        values = tuple(self.answers.values())

        # Build answers dictionary string representation
        answers_str = '{' + f"{keys[0]}: '{values[0]}', ..., {keys[-1]}: '{values[-1]}'" + '}'

        return f'SurveyAnswers({self.student_name}, {answers_str})'
