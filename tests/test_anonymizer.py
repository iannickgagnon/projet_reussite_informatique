import pandas as pd
from survey import Survey
from anonymizer import anonymize, __clean_up

def test_anonymize():

    # Create test data
    course_id = 'INF136'
    semester = 'E2023'
    members = ['John Doe', 'Jane Smith', 'Tom Jones']

    events_data = {'Name': members,
                   'Event': ['EXAM01', 'EXAM02', 'FINAL'],
                   'Grade': ['A', 'B+', 'A-']}

    results_data = {'Name': members,
                    'Score': [90, 75, 85],
                    'Grade': ['A', 'B', 'A-']}

    surveys_data = {'Name': members,
                    'Q1': ['Yes', 'No', 'No'],
                    'Q2': ['No', 'Yes', 'No']}

    events_df = pd.DataFrame(events_data)

    results_df = pd.DataFrame(results_data)

    survey = Survey(surveys_data)

    # Test with events, results, and surveys
    anon_events, anon_results, anon_survey = anonymize(course_id, semester, events_df, results_df, survey)
    assert set(anon_events["Name"].unique()) == {f"{course_id}_{semester}_0", f"{course_id}_{semester}_1", f"{course_id}_{semester}_2"}
    assert set(anon_results["Name"].unique()) == {f"{course_id}_{semester}_0", f"{course_id}_{semester}_1", f"{course_id}_{semester}_2"}
    assert set([survey.student_answers[i].student_name for i in range(len(survey.student_answers))]) == {f"{course_id}_{semester}_0", f"{course_id}_{semester}_1", f"{course_id}_{semester}_2"}

    # Test with only events
    anon_events_only = anonymize(course_id, semester, events_df)
    assert set(anon_events_only["Name"].unique()) == {f"{course_id}_{semester}_0", f"{course_id}_{semester}_1", f"{course_id}_{semester}_2"}

    # Test with only results
    anon_results_only = anonymize(course_id, semester, None, results_df)
    assert set(anon_results_only["Name"].unique()) == {f"{course_id}_{semester}_0", f"{course_id}_{semester}_1", f"{course_id}_{semester}_2"}

    # Test with only survey
    anon_survey_only = anonymize(course_id, semester, None, None, survey)
    assert set([anon_survey_only[2].student_answers[i].student_name for i in range(len(anon_survey_only[2].student_answers))]) == {f"{course_id}_{semester}_0", f"{course_id}_{semester}_1", f"{course_id}_{semester}_2"}

    # Test clean_up
    anon_events, anon_results, anon_survey = anonymize(course_id, semester, events_df, results_df, survey, clean_up=True)
    assert len(anon_events) == 2
    assert len(anon_results) == 2
    assert len(anon_survey.student_answers) == 2

def test_clean_up():
    # Create test data
    prefix = "CS101_Fall2022"
    events_data = {"Name": [f"{prefix}_0", "John Smith"],
                   "Event": ["Homework 1", "Midterm 1"],
                   "Grade": ["A", "B+"]}
    results_data = {"Name": [f"{prefix}_0", "John Smith"],
                    "Score": [90, 75],
                    "Grade": ["A", "C+"]}
    surveys_data = {"Name": ["Jane Doe", f"{prefix}_0", "Tom Jones"],
                    "Q1": ["Yes", "No", "No"],
                    "Q2": ["Sometimes", "Often", "Never"]}
    events_df = pd.DataFrame(events_data)
    results_df = pd.DataFrame(results_data)
    survey = Survey(surveys_data)

    # Test __clean_up
    anon_events, anon_results, anon_survey = __clean_up(prefix, events_df, results_df, survey)
    assert len(anon_events) == 1
    assert len(anon_results) == 1
    assert len(anon_survey.student_answers) == 1
    assert anon_survey.student_answers[0].student_name == f"{prefix}_0"
