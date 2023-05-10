
def letter_grade_to_points(grade: str) -> float:
    """
    Convert a letter grade to a point value.

    Args:
        grade (str): A string representing the letter grade.

    Returns:
        (float): The equivalent point value.

    Raises:
        ValueError: If the input grade is not a valid letter grade.
    """

    if grade == 'A+':
        return 4.3
    elif grade == 'A':
        return 4.0
    elif grade == 'A-':
        return 3.7
    elif grade == 'B+':
        return 3.3
    elif grade == 'B':
        return 3.0
    elif grade == 'B-':
        return 2.7
    elif grade == 'C+':
        return 2.3
    elif grade == 'C':
        return 2.0
    elif grade == 'C-':
        return 1.7
    elif grade == 'D+':
        return 1.3
    elif grade == 'D':
        return 1.0
    elif grade == 'E':
        return 0.0
    else:
        raise ValueError(f'Invalid grade ({grade})')