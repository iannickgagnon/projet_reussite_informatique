
# External libraries
import pickle
from copy import deepcopy
from statistics import mean
import matplotlib.pyplot as plt

# Internal libraries
from survey import Survey
from course import Course
from anonymizer import anonymize
from events_parser import parse_events
from results_parser import parse_results


def analysis_1_a(filename: str,
                 is_anonymize: bool = False,
                 is_regression: bool = False):
    """
    Plots a given course's average grades against engagement with or without a regression line.
    """

    # Get events data
    events_data, course_id, semester_id = parse_events(filename)

    # Get evaluation structure and grades
    evaluation_structure, results_data = parse_results(filename)

    # Anonymize
    if is_anonymize:

        # Get surveys data
        surveys_data = Survey(filename)

        # Invoke anonymizer
        events_data, results_data, surveys_data = anonymize(course_id,
                                                            semester_id,
                                                            events_data,
                                                            results_data,
                                                            surveys_data,
                                                            clean_up=False)
    else:

        # Surveys data is not required if data is not anonymized
        surveys_data = None

    # Build course
    course_obj = Course(evaluation_structure,
                        course_id,
                        semester_id,
                        events_data,
                        results_data,
                        surveys_data)

    # Build and show plot
    fig, ax = course_obj.plot_individual_avg_vs_engagement(normalize=True,
                                                           linear_regression=is_regression)

    # Export figure and axes
    return fig, ax


def analysis_1_b(filename: str,
                 is_anonymize: bool = False):
    """
    Plots a given course's average grades distribution.
    """

    # Get events data
    events_data, course_id, semester_id = parse_events(filename)

    # Get evaluation structure and grades
    evaluation_structure, results_data = parse_results(filename)

    # Anonymize
    if is_anonymize:

        # Get surveys data
        surveys_data = Survey(filename)

        # Invoke anonymizer
        events_data, results_data, surveys_data = anonymize(course_id,
                                                            semester_id,
                                                            events_data,
                                                            results_data,
                                                            surveys_data,
                                                            clean_up=False)
    else:

        # Surveys data is not required if data is not anonymized
        surveys_data = None

    # Build course
    course_obj = Course(evaluation_structure,
                        course_id,
                        semester_id,
                        events_data,
                        results_data,
                        surveys_data)

    # Build and show plot
    fig, ax = course_obj.plot_individual_avg_distribution()

    # Export figure and axes
    return fig, ax


def analysis_1_c(filename: str,
                 is_anonymize: bool = False):
    """
    Plots a given course's average engagement distribution.
    """

    # Get events data
    events_data, course_id, semester_id = parse_events(filename)

    # Get evaluation structure and grades
    evaluation_structure, results_data = parse_results(filename)

    # Anonymize
    if is_anonymize:

        # Get surveys data
        surveys_data = Survey(filename)

        # Invoke anonymizer
        events_data, results_data, surveys_data = anonymize(course_id,
                                                            semester_id,
                                                            events_data,
                                                            results_data,
                                                            surveys_data,
                                                            clean_up=False)
    else:

        # Surveys data is not required if data is not anonymized
        surveys_data = None

    # Build course
    course_obj = Course(evaluation_structure,
                        course_id,
                        semester_id,
                        events_data,
                        results_data,
                        surveys_data)

    # Build and show plot
    fig, ax = course_obj.plot_engagement_distribution()

    # Export figure and axes
    return fig, ax


def analysis_2():
    """
    Plots a list of courses' individual averages against engagement.
    """

    fig, ax = Course.plot_combined_individual_avg_vs_engagement(is_linear_regression=True,
                                                                is_plot_successes=False,
                                                                is_plot_failures=True)

    # Export figure and axes
    return fig, ax


def analysis_3():
    """
    Plots the number of points missing to go from failure to success.
    """

    fig, ax = Course.plot_combined_points_to_pass_vs_engagement(is_linear_regression=True)

    # Export figure and axes
    return fig, ax


def analysis_4():
    """
    Plots the stacked histograms of failures and successes distributions.
    """

    fig, ax = Course.plot_combined_stacked_distributions_pass_fail()

    # Export figure and axes
    return fig, ax


if __name__ == '__main__':

    #analysis_1_a('INF135_02.csv')
    #analysis_1_b('INF135_02.csv')
    #analysis_1_c('INF135_02.csv')
    #analysis_2()
    #analysis_3()
    #analysis_4()

    #courses = Course.build_course_list_from_files()
    #with open('courses.pkl', 'wb') as file:
    #    pickle.dump(courses, file)

    with open('courses.pkl', 'rb') as file:
        courses = pickle.load(file)

    OUTCOMES = {outcome: 0 for outcome in ('Abandon', 'Succès', 'Échec')}

    YES_NO = {'Oui': OUTCOMES.copy(),
              'Non': OUTCOMES.copy()}

    EMPLOI_ETE = {'Études seulement': OUTCOMES.copy(),
                  'Travail seulement': OUTCOMES.copy(),
                  'Études et travail': OUTCOMES.copy()}

    LANGUE = {'Français': OUTCOMES.copy(),
              'Anglais': OUTCOMES.copy(),
              'Autre': OUTCOMES.copy()}

    SITUATION_FINANCIERE = {'Aisée': OUTCOMES.copy(),
                            'Satisfaisante': OUTCOMES.copy(),
                            'Précaire': OUTCOMES.copy()}

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

    NUMERICAL_QUESTIONS_INDEX = (6,)

    # Analyser tous les sigles confondus
    is_found = 0
    for course in courses:
        for answers in course.surveys:
            for student in course.students:
                if student.name == answers.student_name:
                    for question_index, answer in enumerate(answers):
                        if question_index in NUMERICAL_QUESTIONS_INDEX:
                            questions_and_outcomes[question_index].append(answers[question_index])
                        else:
                            questions_and_outcomes[question_index][answer][student.get_outcome()] += 1





    # Print results
    nb_responses = len(questions_and_outcomes[6])
    for question_index in questions_and_outcomes:
        print(f'Question no.{question_index + 1}\n')
        if question_index == 6:
            print(f'\t\tMoyenne - {mean(questions_and_outcomes[question_index]):.1f} heures travaillées\n')
        else:
            for answer in questions_and_outcomes[question_index]:

                freq_answer = sum(questions_and_outcomes[question_index][answer].values())

                nb_abandon = questions_and_outcomes[question_index][answer]['Abandon']
                nb_succes = questions_and_outcomes[question_index][answer]['Succès']
                nb_echec = questions_and_outcomes[question_index][answer]['Échec']

                total = nb_abandon + nb_succes + nb_echec

                if freq_answer == 0.0:
                    pct_abandon = 0.0
                    pct_succes = 0.0
                    pct_echec = 0.0
                else:
                    pct_abandon = nb_abandon / freq_answer * 100
                    pct_succes = nb_succes / freq_answer * 100
                    pct_echec = nb_echec / freq_answer * 100

                print(f'\t{answer} - {(total / nb_responses * 100):.1f}% ({total})\n')

                print(f'\t\tAbandon - {pct_abandon:.1f}% ({nb_abandon})')
                print(f'\t\tSuccès  - {pct_succes:.1f}% ({nb_succes})')
                print(f'\t\tÉchec   - {pct_echec:.1f}% ({nb_echec})\n')
