
# External libraries
import os
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

# Internal libraries
import survey
from student import Student
from anonymizer import anonymize
from events_parser import parse_events
from results_parser import parse_results

# Internal constants
from constants import (
    _0_PERCENT,
    _100_PERCENT,
    _50_PERCENT,
    _0_EVENTS,
    COL_NAME_NAME,
    COL_NAME_AVERAGE,
    COL_NAME_GRADE,
    IS_GROUP_PROJECT,
    ROW_NAME_WEIGHT,
    ROW_NAME_GROUP_PROJECT,
    PATH_MAIN_PLOT_STYLE,
    LAST_ELEMENT_INDEX,
    PATH_EVENTS,
    PATH_RESULTS,
    PATH_SURVEYS,
    PASSING_GRADE,
    PLOT_X_LABEL_ENGAGEMENT,
    PLOT_Y_LABEL_INDIVIDUAL_AVERAGE,
    PLOT_Y_LABEL_COUNT,
    PLOT_LABEL_FREQUENCY,
    PLOT_BORDER_MARGIN_FACTOR,
    PLOT_BORDER_MARGIN_FACTOR_SMALL,
    PLOT_HISTOGRAM_CONFIG,
)


class Course:
    """
    A class representing a course.

    Attributes:
        evaluation_structure (DataFrame): The evaluation structure of the course.
        course_id (str): The ID of the course.
        semester_id (str): The ID of the semester.
        events (DataFrame): The raw data of the events.
        results (DataFrame): The raw data of the results.
        surveys (Survey): The raw data of the surveys.
        nb_students (int): The number of students in the course.
        nb_events (int): The number of events in the course.
        students (list): The list of students in the course.

    Methods:

        __init__(self, evaluation_structure: DataFrame, course_id: str, semester_id: str, events: DataFrame,
                 results: DataFrame, surveys: survey.Survey) -> None:
            Initializes a new instance of the Course class.

        __build(self) -> None:
            Builds the data structure of the course.

        get_student_events_by_name(self, name: str):
            Gets the events of a specific student.

        get_student_results_by_name(self, name: str):
            Gets the results of a specific student.

        __student_calculate_averages(self, student: Student):
            Calculates the averages of a specific student.

        __get_individual_avg_vector(self, normalize=False):
            Gets the vector of individual averages.

        __get_engagement_vector(self, normalize=False):
            Gets the vector of student engagement.

        plot_individual_avg_vs_engagement(self, normalize=False, linear_regression=False):
            Plots individual average vs engagement.

        build_course_list_from_files(path_events_files: str = PATH_EVENTS, path_results_files: str = PATH_RESULTS,
                                     path_results_surveys: str = PATH_SURVEYS):
            Builds a list of courses from files.

        split_courses_fail_pass(courses):
            Separates courses into failures and successes.

        plot_combined_individual_avg_vs_engagement(linear_regression=False):
            Plots the combined individual average vs engagement.

        plot_combined_points_to_pass_vs_engagement(linear_regression=False):
            Plots the combined points to pass vs engagement.

        plot_combined_stacked_distributions_pass_fail():
            Plots the combined stacked distributions for passes and failures.

        letter_grade_to_points(grade: str) -> float:
            Converts a letter grade to a points value.
    """

    def __init__(self,
                 evaluation_structure: pd.DataFrame,
                 course_id: str,
                 semester_id: str,
                 events: pd.DataFrame,
                 results: pd.DataFrame,
                 surveys: survey.Survey = None) -> None:
        """
        Initializes a new instance of the Course class.

        Args:
            evaluation_structure (DataFrame): The evaluation structure of the course.
            course_id (str): The ID of the course.
            semester_id (str): The ID of the semester the course is taught in.
            events (DataFrame): The events associated with the course.
            results (DataFrame): The results of the course.
            surveys (survey.Survey, optional): The survey responses for the course. Defaults to None.
        """

        # Store raw data
        self.events = events
        self.results = results
        self.surveys = surveys
        self.evaluation_structure = evaluation_structure
        self.course_id = course_id
        self.semester_id = semester_id

        # Store dimensions
        self.nb_students = results.shape[0]
        self.nb_events = events.shape[0]

        # Initialize students list
        self.students = []

        # Build data structure
        self.__build()

    def __build(self) -> None:
        """
        Builds the data structure for the course by iterating over the internal students list, creating a Student
        object for each, calculating and storing their averages and letter grades.

        Returns:
            None
        """

        for i in range(self.nb_students):

            # Extract name
            name = self.results.iloc[i][COL_NAME_NAME]

            # Extract student's data
            events = self.get_student_events_by_name(name)
            results = self.get_student_results_by_name(name)

            # Add student to pool
            self.students.append(Student(name, events, results))

            # Calculate student's averages
            self.__student_calculate_averages(self.students[LAST_ELEMENT_INDEX])
            self.students[LAST_ELEMENT_INDEX].overall_average = results[COL_NAME_AVERAGE].iloc[0]

            # Store student's letter grades
            self.students[LAST_ELEMENT_INDEX].grade = results[COL_NAME_GRADE].iat[0]

    def get_student_events_by_name(self, name: str):
        """
        Gets the events associated with a student by their name.

        Args:
            name (str): The name of the student.

        Returns:
            (DataFrame): The events associated with the student.
        """

        return self.events.query(f'{COL_NAME_NAME} == \'{name}\'')

    def get_student_results_by_name(self, name: str):
        """
        Gets the results of a student by their name.

        Args:
            name (str): The name of the student.

        Returns:
            (DataFrame): The results of the student.
        """
        return self.results.query(f'{COL_NAME_NAME} == \'{name}\'')

    def __student_calculate_averages(self, student: Student):
        """
        Calculates individual and group work averages for a given student based on the
        evaluation structure and the student's results.

        Args:
            student (Student): The student to calculate the averages for.

        Returns:
            None
        """

        # Initialize student's averages
        student.individual_average = 0.0
        student.group_work_average = 0.0

        # Initialize weights for individual and group evaluations
        weight_individual = 0.0
        weight_group_projects = 0.0

        # Go through all evaluated elements
        for col in self.evaluation_structure.columns:

            # Extract current evaluation element's weight
            weight = self.evaluation_structure.loc[ROW_NAME_WEIGHT].at[col]

            if self.evaluation_structure.loc[ROW_NAME_GROUP_PROJECT].at[col] == IS_GROUP_PROJECT:
                weight_group_projects += weight
                if not pd.isna(student.results[col].iloc[0]):
                    student.group_work_average += weight * student.results[col].iloc[0]
            else:
                weight_individual += weight
                if not pd.isna(student.results[col].iloc[0]):
                    student.individual_average += weight * student.results[col].iloc[0]

        # Normalize for weight
        student.individual_average /= weight_individual
        student.group_work_average /= weight_group_projects

        # Make sure weights sum to 100 percent
        assert weight_individual + weight_group_projects == _100_PERCENT, \
            f'Sum of weights not equal to 100 {weight_individual + weight_group_projects}'

    def __get_individual_avg_vector(self, normalize=False):
        """
        Gets the individual averages of all the students in the course.

        Args:
            normalize (bool): Whether to normalize the output or not.

        Returns:
            (List[float]): The individual averages vector.
        """

        # TODO: list comprehension after unit tests are written
        average = []
        for student in self.students:
            average.append(student.individual_average)

        if normalize:
            average = np.array(average) / max(average)

        return average

    def __get_engagement_vector(self, normalize=False):
        """
        Gets the number of events attended by each student in the course.

        Args:
            normalize (bool): Whether to normalize the engagement vector.

        Returns:
            (List[int]): The engagement vector.
        """

        # TODO: list comprehension after unit tests are written
        nb_events = []
        for student in self.students:
            nb_events.append(student.nb_events)

        if normalize:
            nb_events = np.array(nb_events) / max(nb_events)

        return nb_events

    def plot_individual_avg_vs_engagement(self, normalize=False, linear_regression=False):
        """
        Builds a scatter plot of the individual average vs engagement vector of the students in the course.

        Args:
            normalize (bool): Whether to normalize the engagement vector or not.
            linear_regression (bool): Whether to plot a linear regression line or not.

        Returns:
            (Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]): The figure and axes objects of the plot.
        """

        # Get events counts and individual average
        average = self.__get_individual_avg_vector()
        nb_events = self.__get_engagement_vector(normalize=normalize)

        # Plot
        with plt.style.context(PATH_MAIN_PLOT_STYLE):

            # Create figure and axes
            fig, ax = plt.subplots()

            # Add scatterplot
            ax.scatter(nb_events, average, color='blue')

            # Add linear regression
            if linear_regression:

                # Create model
                model = LinearRegression()
                model.fit(np.array(nb_events).reshape(-1, 1), average)

                # Get model output
                x_regression = np.linspace(0, 1, 2)
                y_regression = model.predict(x_regression[:, np.newaxis])

                # Add to plot
                ax.plot(x_regression, y_regression, '--')

            # Labels
            plt.xlabel(PLOT_X_LABEL_ENGAGEMENT)
            plt.ylabel(PLOT_Y_LABEL_INDIVIDUAL_AVERAGE)

            # Limits
            plt.xlim(_0_EVENTS, max(nb_events) * PLOT_BORDER_MARGIN_FACTOR)
            plt.ylim(_0_PERCENT, _100_PERCENT)

            # Show plot
            plt.show()

            # Export figure and axes
            return fig, ax

    def plot_individual_avg_distribution(self):
        """
        Builds a histogram of individual averages.

        Args:
            None.

        Returns:
            (Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]): The figure and axes objects of the plot.
        """

        # Get events counts and individual average
        averages = self.__get_individual_avg_vector()

        # Plot
        with plt.style.context(PATH_MAIN_PLOT_STYLE):
            fig, ax = plt.subplots()

            # Build histogram
            _, bins, _ = ax.hist(averages,
                                 color='royalblue',
                                 edgecolor='k',
                                 density=1)

            # Add density estimate
            density = gaussian_kde(averages)
            density_x = np.linspace(bins[0], bins[-1], 50)
            density_y = density.evaluate(density_x)
            ax.plot(density_x, density_y, '--r')

            # Show corresponding normal distributions
            mu = np.mean(averages)
            sigma = np.std(averages)
            y_normal = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
            ax.plot(bins, y_normal, '--', color='k')

            # Labels
            plt.xlabel(PLOT_Y_LABEL_INDIVIDUAL_AVERAGE)
            plt.ylabel(PLOT_LABEL_FREQUENCY)

            # Add legend
            plt.legend(('Density est.', 'Normal ref.'))

            # Show plot
            plt.show()

            # Export figure and axes
            return fig, ax

    def plot_engagement_distribution(self):
        """
        Builds a histogram of individual averages.

        Args:
            None.

        Returns:
            (Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]): The figure and axes objects of the plot.
        """

        # Get events counts and individual average
        engagement = self.__get_engagement_vector()

        # Plot
        with plt.style.context(PATH_MAIN_PLOT_STYLE):
            fig, ax = plt.subplots()

            # Build histogram
            _, bins, _ = ax.hist(engagement,
                                 color='royalblue',
                                 edgecolor='k',
                                 density=1)

            # Add density estimate
            density = gaussian_kde(engagement)
            density_x = np.linspace(bins[0], bins[-1], 50)
            density_y = density.evaluate(density_x)
            ax.plot(density_x, density_y, '--r')

            # Show corresponding normal distributions
            mu = np.mean(engagement)
            sigma = np.std(engagement)
            y_normal = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
            ax.plot(bins, y_normal, '--', color='k')

            # Labels
            plt.xlabel(PLOT_X_LABEL_ENGAGEMENT)
            plt.ylabel(PLOT_LABEL_FREQUENCY)

            # Add legend
            plt.legend(('Density est.', 'Normal ref.'))

            # Show plot
            plt.show()

            # Export figure and axes
            return fig, ax

    @staticmethod
    def build_course_list_from_files(path_events_files: str = PATH_EVENTS,
                                     path_results_files: str = PATH_RESULTS,
                                     path_results_surveys: str = PATH_SURVEYS):
        """
        Builds a list of Course objects from event and results files.

        Args:
            path_events_files (str): The path to the events files.
            path_results_files (str): The path to the results files.
            path_results_surveys (str): The path to the surveys files.

        Returns:
              List[Course]: A list of Course objects built from the provided files.
    """

        # Get file names
        filenames_events = os.listdir(path_events_files)
        filenames_results = os.listdir(path_results_files)
        filenames_surveys = os.listdir(path_results_surveys)

        course_list = []
        for filename_results in filenames_results:

            # Extract course and group identifiers
            course_tag, group_tag = filename_results[:9].split('_')

            # Find corresponding events file
            try:
                i = 0
                while not ((course_tag + '_') in filenames_events[i] and ('_' + group_tag) in filenames_events[i]):
                    i += 1
            except IndexError:
                raise IndexError(f'Could not find events file for {course_tag}-{group_tag}')

            # Parse
            events_data, course_id, semester_id = parse_events(filenames_events[i])
            evaluation_structure, results_data = parse_results(filename_results)

            # FIXME: This won't work because files are not synced
            surveys_data = survey.Survey(filenames_surveys[i])

            # Anonymize
            events_data, results_data, surveys_data = anonymize(course_id,
                                                                semester_id,
                                                                events=events_data,
                                                                results=results_data,
                                                                surveys=surveys_data,
                                                                clean_up=False)

            # Build current course
            current_course = Course(evaluation_structure,
                                    course_id,
                                    semester_id,
                                    events=events_data,
                                    results=results_data,
                                    surveys=surveys_data)

            # Store
            course_list.append(current_course)

        return course_list

    @staticmethod
    def split_courses_fail_pass(courses):
        """
        Separates students who pass and fail from the provided list of courses based on their individual average.

        Args:
            courses (List[Course]): The list of courses to separate.

        Returns:
            (Tuple[List[float], List[int], List[float], List[int]]): Failing and succeeding averages with counts.
        """

        # Initialize containers to separate failures from successes
        averages_pass = []
        nb_events_pass = []
        averages_fail = []
        nb_events_fail = []

        for course in courses:

            # Get events counts and individual averages
            course_averages = course.__get_individual_avg_vector()
            course_nb_events = course.__get_engagement_vector(normalize=True)

            for i in range(len(course_averages)):

                # Separate failures from successes
                if course_averages[i] >= PASSING_GRADE:
                    averages_pass.append(course_averages[i])
                    nb_events_pass.append(course_nb_events[i])
                else:
                    averages_fail.append(course_averages[i])
                    nb_events_fail.append(course_nb_events[i])

        return averages_pass, nb_events_pass, averages_fail, nb_events_fail

    @staticmethod
    def plot_combined_individual_avg_vs_engagement(is_linear_regression: bool = False,
                                                   is_plot_successes: bool = True,
                                                   is_plot_failures: bool = True) \
            -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """
        Plots a combined scatter plot of the individual average vs engagement vector for all courses,
        with passing and failing students plotted separately.

        Args:
            is_linear_regression (bool): Whether to plot regression line for failures. Defaults to False.
            is_plot_successes (bool): Whether to plot students who succeeded. Defaults to True.
            is_plot_failures (bool): Whether to plot students who failed. Defaults to True.

        Returns:
            (Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]): The figure and axes objects of the plot.
        """

        # Make sure there is something to plot
        assert is_plot_successes or is_plot_failures, 'Must plot either successes, failures or both'

        # Build course list
        # course_list = Course.build_course_list_from_files()

        # TODO: Remove
        import pickle
        with open('courses.pkl', 'rb') as file:
            course_list = pickle.load(file)

        # Split failures and successes
        averages_success, nb_events_success, averages_fail, nb_events_fail = \
            Course.split_courses_fail_pass(course_list)

        # Plot
        with plt.style.context(PATH_MAIN_PLOT_STYLE):

            # Create figure and axes
            fig, ax = plt.subplots()

            # Plot successes
            if is_plot_successes:
                ax.scatter(nb_events_success, averages_success, color='royalblue', s=4)

            # Plot failures
            if is_plot_failures:
                ax.scatter(nb_events_fail, averages_fail, color='indianred', s=4)

            # Build data vectors based on regression options
            if is_plot_successes and is_plot_failures:
                # Combine failures and successes
                nb_events_fail.extend(nb_events_success)
                averages_fail.extend(averages_success)
                data_x = np.array(nb_events_fail).reshape(-1, 1)
                data_y = np.array(averages_fail).reshape(-1, 1)
            elif is_plot_successes:
                data_x = np.array(nb_events_success).reshape(-1, 1)
                data_y = np.array(averages_success).reshape(-1, 1)
            else:
                data_x = np.array(nb_events_fail).reshape(-1, 1)
                data_y = np.array(averages_fail).reshape(-1, 1)

            # Linear regression for failures
            if is_linear_regression:

                # Create model
                model = LinearRegression()
                model.loss = 'mae'

                # Fit model to data
                model.fit(data_x, data_y)

                # Find min/max for x-axis
                x_min = nb_events_fail[np.argmin(nb_events_fail)]
                x_max = nb_events_fail[np.argmax(nb_events_fail)]

                # Model output
                x_regression = np.linspace(x_min, x_max, 100)
                y_regression = model.predict(x_regression[:, np.newaxis])

                # Add to plot
                ax.plot(x_regression, y_regression, 'k--')

            # Labels
            plt.xlabel(PLOT_X_LABEL_ENGAGEMENT)
            plt.ylabel(PLOT_Y_LABEL_INDIVIDUAL_AVERAGE)

            # Limits
            plt.xlim(_0_EVENTS, max(nb_events_success + nb_events_fail) * PLOT_BORDER_MARGIN_FACTOR_SMALL)

            if is_plot_successes and is_plot_failures:
                # Vertical axis ranges from 0 to 100 when both successes and failures are plotted
                plt.ylim(_0_PERCENT, _100_PERCENT * PLOT_BORDER_MARGIN_FACTOR_SMALL)
                legend_items = ['Pass', 'Fail']
            elif is_plot_successes:
                # Vertical axis ranges from 50 to 100 when only successes are plotted
                plt.ylim(_50_PERCENT, _100_PERCENT * PLOT_BORDER_MARGIN_FACTOR_SMALL)
                legend_items = ['Pass']
            else:
                # Vertical axis ranges from 0 to 50 when only failures are plotted
                plt.ylim(_0_PERCENT, _50_PERCENT * PLOT_BORDER_MARGIN_FACTOR_SMALL)
                legend_items = ['Fail']

            # Specify MAE regression in title
            if is_linear_regression:
                plt.title('Linear regression using Mean Absolute Error')

            # Add legend
            plt.legend(legend_items, loc='upper left')

        # Show plot
        plt.show()

        # Print coefficients if regression is done
        if is_linear_regression:
            print(f'Slope     : {model.coef_[0][0]:.2f}')
            print(f'Intercept : {model.intercept_[0]:.2f}')

        # Export figure and axes
        return fig, ax

    @staticmethod
    def plot_combined_points_to_pass_vs_engagement(is_linear_regression=False):
        """
        Plots a scatter plot of the number of points missing to pass vs engagement vector for failing students.

        Args:
            is_linear_regression (bool): Whether to plot regression line for the number of points missing to pass.

        Returns:
            (Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]): The figure and axes objects of the plot.
        """

        # Extract courses list
        #courses = Course.build_course_list_from_files()

        # TODO: Remove
        import pickle
        with open('courses.pkl', 'rb') as file:
            courses = pickle.load(file)

        # Split failures from successes
        averages_pass, nb_events_pass, averages_fail, nb_events_fail = Course.split_courses_fail_pass(courses)

        # Calculate the number of points missing to pass
        delta_to_pass = np.abs(np.array(averages_fail) - PASSING_GRADE)

        # Plot
        with plt.style.context('./images/main_plot_style.mplstyle'):

            # Create figure and axes
            fig, ax = plt.subplots()

            # Build scatterplot
            ax.scatter(nb_events_fail, delta_to_pass, color='blue', label='_nolegend_')

            if is_linear_regression:

                # Create model
                model = LinearRegression()
                model.loss = 'mae'

                # Fit model to data
                model.fit(np.array(nb_events_fail).reshape(-1, 1), delta_to_pass)

                # Model output
                x_regression = np.linspace(0, 1, 100)
                y_regression = model.predict(x_regression[:, np.newaxis])

                # Add to plot
                ax.plot(x_regression, y_regression, 'k--')

        # Labels
        plt.xlabel('Engagement')
        plt.ylabel('Points to pass')

        # Limits
        plt.xlim(0, max(nb_events_fail))
        plt.ylim(-1, PASSING_GRADE + 1)

        # Show plot
        plt.show()

        # Print coefficients if regression is done
        if is_linear_regression:
            print(f'Slope     : {model.coef_[0]:.2f}')
            print(f'Intercept : {model.intercept_:.2f}')

        # Export figure and axes
        return fig, ax

    @staticmethod
    def plot_combined_stacked_distributions_pass_fail():
        """
        Plots a stacked histogram of the engagement vector for passing and failing students.

        Returns:
            (Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]): The figure and axes objects of the plot.
        """

        # Extract courses list
        #courses = Course.build_course_list_from_files()

        # TODO: Remove
        import pickle
        with open('courses.pkl', 'rb') as file:
            courses = pickle.load(file)

        # Split failures from successes
        averages_pass, nb_events_pass, averages_fail, nb_events_fail = Course.split_courses_fail_pass(courses)

        # Combine averages to get overall distribution
        averages_all = averages_pass.copy()
        averages_all.extend(averages_fail)

        # Combine events to get overall distribution
        nb_events_all = nb_events_pass.copy()
        nb_events_all.extend(nb_events_fail)

        # Plot
        with plt.style.context(PATH_MAIN_PLOT_STYLE):

            # Create figure and axes
            fig, ax = plt.subplots()

            # Create histograms
            counts_all, _, _ = ax.hist(nb_events_all, color='royalblue', **PLOT_HISTOGRAM_CONFIG)
            counts_fail, _, _ = ax.hist(nb_events_fail, color='indianred', **PLOT_HISTOGRAM_CONFIG)

        # Labels
        plt.xlabel(PLOT_X_LABEL_ENGAGEMENT)
        plt.ylabel(PLOT_Y_LABEL_COUNT)

        # Add legend
        plt.legend(['Pass', 'Fail'])

        # Show plot
        plt.show()

        # Export figure and axes
        return fig, ax
