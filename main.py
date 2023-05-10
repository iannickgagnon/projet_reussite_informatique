
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
    '''
    Plots a given course's average grades against engagement with or without a regression line.
    '''

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


def analysis_1_b(filename: str,
                 is_anonymize: bool = False):
    '''
    Plots a given course's average grades distribution.
    '''

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


def analysis_1_c(filename: str,
                 is_anonymize: bool = False):
    '''
    Plots a given course's average engagement distribution.
    '''

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


def analysis_2():
    '''
    Plots a list of courses' individual averages against engagement.
    '''

    fig, ax = Course.plot_combined_individual_avg_vs_engagement(is_linear_regression=True,
                                                                is_plot_successes=False,
                                                                is_plot_failures=True)


def analysis_3():
    '''
    Plots the number of points missing to go from failure to success.
    '''

    fig, ax = Course.plot_combined_points_to_pass_vs_engagement(is_linear_regression=True)









def analysis_4():

    # Analysis 4
    fig, ax = Course.plot_combined_stacked_distributions_pass_fail()


def analysis_5():

    '''
    # Extract courses list
    courses = Course.build_course_list_from_files()

    with open('mock_courses.pkl', 'wb') as file:
        pickle.dump(courses, file)
    '''

    with open('mock_courses.pkl', 'rb') as file:
        courses = pickle.load(file)

    import numpy as np
    from sklearn.linear_model import LinearRegression

    slopes = []

    regressor = LinearRegression()

    for course in courses:
        for student in course.students:

            results = student.get_exam_results().iloc[0]

            regressor.fit(np.arange(len(results)).reshape(-1,1), results)

            slopes.append(regressor.coef_[0])

            #plt.plot(results)


    plt.hist(slopes)
    plt.show()

    pass



if __name__ == '__main__':

    #analysis_1_a('INF135_02.csv')
    #analysis_1_b('INF135_02.csv')
    #analysis_1_c('INF135_02.csv')

    #analysis_2()

    #analysis_3()

    #TODO: Add listdir and fix _build_from_list() at the same time
    course_identifiers = ('INF111', 'INF130', 'INF135', 'INF147', 'INF155')




    '''

    # Build courses container (dict)
    courses = {course_id:[] for course_id in course_identifiers}

    # Get filenames
    filenames_events = listdir(PATH_EVENTS)
    filenames_results = listdir(PATH_RESULTS)
    filenames_surveys = listdir(PATH_SURVEYS)

    for filename in filenames_events:

        # Make sure the corresponding results and surveys are found
        assert filename in filenames_results, f'Results file not found ({filename})'
        assert filename in filenames_surveys, f'Surveys file not found ({filename})'

        # Get events data
        events_data, course_id, semester_id = parse_events(f'{PATH_EVENTS}{filename}')

        # Get evaluation structure and grades
        evaluation_structure, results_data = parse_results(f'{PATH_RESULTS}{filename}')

        # Get survey data
        surveys_data = Survey(f'{PATH_SURVEYS}{filename}')

        # Anonymize
        events_data, results_data, surveys_data = anonymize(course_id,
                                                            semester_id,
                                                            events=events_data,
                                                            results=results_data,
                                                            surveys=surveys_data,
                                                            clean_up=True)

        # Build course
        course_obj = Course(evaluation_structure,
                            course_id,
                            semester_id,
                            events=events_data,
                            results=results_data,
                            surveys=surveys_data)

        # Add to courses dictionary
        courses[filename[:COURSE_ID_LENGTH]].append(course_obj)
    '''

    '''
    with open('courses_dict.pkl', 'bw') as file:
        pickle.dump(courses, file)
    '''

    with open('courses_dict.pkl', 'rb') as file:
        courses = pickle.load(file)

    OUTCOMES = {outcome:0 for outcome in ('Abandon', 'Succès', 'Échec')}

    YES_NO = {'Oui':OUTCOMES.copy(),
              'Non':OUTCOMES.copy()}

    EMPLOI_ETE = {'Études seulement':OUTCOMES.copy(),
                  'Travail seulement':OUTCOMES.copy(),
                  'Études et travail':OUTCOMES.copy()}

    LANGUE = {'Français':OUTCOMES.copy(),
              'Anglais':OUTCOMES.copy(),
              'Autre':OUTCOMES.copy()}

    SITUATION_FINANCIERE = {'Aisée':OUTCOMES.copy(),
                            'Satisfaisante':OUTCOMES.copy(),
                            'Précaire':OUTCOMES.copy()}

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
    for course_id, course in courses.items():
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

                pct_abandon = nb_abandon / freq_answer * 100
                pct_succes = nb_succes / freq_answer * 100
                pct_echec = nb_echec / freq_answer * 100

                print(f'\t{answer} - {(total / nb_responses * 100):.1f}% ({total})\n')

                print(f'\t\tAbandon - {pct_abandon:.1f}% ({nb_abandon})')
                print(f'\t\tSuccès  - {pct_succes:.1f}% ({nb_succes})')
                print(f'\t\tÉchec   - {pct_echec:.1f}% ({nb_echec})\n')
