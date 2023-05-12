
# Paths to project files
PATH_EVENTS = './data/events/'
PATH_RESULTS = './data/results/'
PATH_SURVEYS = './data/surveys/'
PATH_TEST_SURVEY_FROM_TESTS = './test_data/test_survey.csv'
PATH_TEST_RESULTS_FROM_TESTS = './test_data/test_results.csv'
PATH_TEST_EVENTS_FROM_TESTS = './test_data/test_events.csv'
PATH_MAIN_PLOT_STYLE = './images/main_plot_style.mplstyle'

# DataFrame column names

ORIGINAL_COL_NAME_NAME = 'Nom complet'
ORIGINAL_COL_NAME_TIME = 'Heure'

COL_NAME_EVENT = 'Event'
COL_NAME_AVERAGE = 'Average'
COL_NAME_GRADE = 'Grade'
COL_NAME_EXAM = 'EXAM'
COL_NAME_FINAL = 'FINAL'
COL_NAME_NAME = 'Name'
COL_NAME_DATE = 'Date'
COL_NAME_TIME = 'Time'
COL_NAME_CONTEXT = 'Contexte de l\'événement'

COLS_TO_REMOVE = ('Heure',
                  'Utilisateur concerné',
                  'Contexte de l\'événement',
                  'Composant',
                  'Description',
                  'Origine',
                  'Adresse IP')

COLS_TO_RENAME = {'Nom complet': 'Name',
                  'Nom de l\'événement': 'Event'}

# DataFrame row names
# Row indexes
ROW_NAME_GROUP_PROJECT = 'Group project'
ROW_NAME_WEIGHT = 'Weight'

# Outcome strings
OUTCOME_ECHEC = 'Échec'
OUTCOME_ABANDON = 'Abandon'
OUTCOME_SUCCES = 'Succès'

# Grades associated to outcomes
OUTCOME_GRADE_ECHEC = 'E'
OUTCOME_GRADES_ABANDON = ('XX', 'AX')

# Constants related to surveys
SURVEY_NB_QUESTIONS = 12
SURVEY_LINE_LENGTH_COMPLETE = 28
SURVEY_LINE_ANSWER_Q1 = 14
SURVEY_LINE_FIRST_NAME = 26
SURVEY_LINE_LAST_NAME = 27
SURVEY_NO_QUESTION_NB_WORK_HOURS = 6

# General
COURSE_ID_LENGTH = 6

# Letter grades
LETTER_GRADE_E = 'E'
LETTER_GRADE_D = 'D'
LETTER_GRADE_D_MINUS = 'D-'
LETTER_GRADE_D_PLUS = 'D+'
LETTER_GRADE_C_MINUS = 'C-'
LETTER_GRADE_C = 'C'
LETTER_GRADE_C_PLUS = 'C+'
LETTER_GRADE_B_MINUS = 'B-'
LETTER_GRADE_B = 'B'
LETTER_GRADE_B_PLUS = 'B+'
LETTER_GRADE_A_MINUS = 'A-'
LETTER_GRADE_A = 'A'
LETTER_GRADE_A_PLUS = 'A+'

# Constants related to timestamps
TIMESTAMP_TOKENS_TO_REPLACE = ' .:-'

# Percentages
_0_PERCENT = 0.0
_100_PERCENT = 100.0
_50_PERCENT = 50.0

# Minimum grade to pass
PASSING_GRADE = 50.0

# Flag for group projects in evaluation structure
IS_GROUP_PROJECT = 1.0

# Number of events
_0_EVENTS = 0

# Last element of an indexed iterable
LAST_ELEMENT_INDEX = -1

# Plot labels
PLOT_X_LABEL_ENGAGEMENT = 'Engagement'
PLOT_Y_LABEL_INDIVIDUAL_AVERAGE = 'Individual average [%]'
PLOT_Y_LABEL_COUNT = 'Count'
PLOT_LABEL_FREQUENCY = 'Frequency'

# Plot border margins
PLOT_BORDER_MARGIN_FACTOR = 1.1
PLOT_BORDER_MARGIN_FACTOR_SMALL = 1.025

# Histogram style
PLOT_HISTOGRAM_CONFIG = {'edgecolor': 'black',
                         'bins': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                         'linewidth': 0.5}

# Course outcomes
COURSE_OUTCOMES = ('Abandon', 'Succès', 'Échec')
COURSE_NB_OUTCOMES = len(COURSE_OUTCOMES)
