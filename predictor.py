
# External libraries
import warnings
import numpy as np
import pandas as pd
from numpy import ravel
from typing import List
from typing import Union
from typing import Tuple
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Internal libraries
from tools import bootstrap_calculate_confidence_interval


def train_and_test_model(model: Union[SVC, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier],
                         x_train: pd.DataFrame,
                         y_train: pd.DataFrame,
                         x_test: pd.DataFrame) \
                         -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fits a given model to training data and tests it on a given test set.

    Args:
        model (Union[SVC, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]): A classifier model.
        x_train (pd.DataFrame): Training predictor data.
        y_train (pd.DataFrame): Training response data.
        x_test (pd.DataFrame): Test predictor data.

    Returns:
        y_train_pred (pd.DataFrame): Predicted response on training data.
        y_test_pred (pd.DataFrame): Predicted response on test data.
    """

    # Train model
    model.fit(x_train, ravel(y_train))

    # Make predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    return y_train_pred, y_test_pred


def calculate_performance_metrics(y_train: pd.DataFrame,
                                  y_train_pred: pd.DataFrame,
                                  y_test: pd.DataFrame,
                                  y_test_pred: pd.DataFrame,
                                  labels: List[str] = None,
                                  verbose: bool = False):
    """
    Calculates accuracy and precision on training and test data.

    Args:
        y_train (pd.DataFrame): Training response data.
        y_train_pred (pd.DataFrame): Training predicted response data.
        y_test (pd.DataFrame): Test response data.
        y_test_pred (pd.DataFrame): Test predicted response data.
        labels (List[str]): List of labels for outcomes.
        verbose (bool): Printout flag.

    Returns:
        train_accuracy (float): Model accuracy on the training set.
        test_accuracy (float): Model accuracy on the test set.
        train_precision (float): Model precision on the training set.
        test_precision (float): Model precision on the test set.
    """

    # FIXME: Unused

    # Calculate training performance metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average=None)

    # Calculate test performance metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average=None)

    # Show performance metrics
    if verbose:
        print("\tTraining Accuracy  :", train_accuracy)
        print("\tTest Accuracy      :", test_accuracy)
        print("\n\tTraining Precision :", ' '.join([f'{label}: {precision:.2f}\t' for label, precision in zip(labels, train_precision)]))
        print("\tTest Precision     :", ' '.join([f'{label}: {precision:.2f}\t' for label, precision in zip(labels, test_precision)]))

    return train_accuracy, test_accuracy, train_precision, test_precision


def encode_full_data(course_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.array]:
    """
    Encore the entire course dataset.

    Args:
        course_data (pd.DataFrame): The course's entire dataset.

    Returns:
        x_data (pd.DataFrame): Predictor variables data.
        y_data (pd.DataFrame): Response variable data.
        classes (np.ndarray): Encoded class names.
    """

    # One-hot encode predictors
    x_data = pd.get_dummies(course_data.iloc[:, :-1])

    # Label encode response
    y_encoder = LabelEncoder()
    y_data = pd.DataFrame({'Outcome': y_encoder.fit_transform(course_data.iloc[:, -1])})

    # Extract encoded classes
    classes = y_encoder.classes_

    return x_data, y_data, classes


def balance_classes(x_data: pd.DataFrame, y_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Balances data by oversampling the underrepresented classes using the SMOTE technique.

    Args:
        x_data (pd.DataFrame): Predictor variables data.
        y_data (pd.DataFrame): Response variable data.

    Returns:
        x_data_resampled (pd.DataFrame): Balanced predictor variables data.
        y_data_resampled (pd.DataFrame): Balanced response variable data.
    """

    # Initialize Synthetic Minority Oversampling TEchnique (SMOTE) model
    smote = SMOTE()

    # Balance classes
    x_data_resampled, y_data_resampled = smote.fit_resample(x_data, y_data)

    return x_data_resampled, y_data_resampled


def run_model_and_evaluate(model: Union[LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier],
                           model_name: str,
                           x_data: pd.DataFrame,
                           y_data: pd.DataFrame,
                           classes: np.ndarray,
                           nb_bootstrap_samples: int = 1000) -> None:
    """
    Runs the model and evaluates it repetitively to generate confidence intervals for each performance metric.

    Args:
        model (Union[LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier]): A classifier.
        model_name (str): The name of the model.
        x_data (pd.DataFrame): Predictor variables data.
        y_data (pd.DataFrame): Response variable data.
        classes (np.ndarray): Encoded response classes.
        nb_bootstrap_samples (int, optional):  Resampling count. Defaults to 10^3.

    Returns:
        Nothing.
    """

    # TODO: Simplify and refactor

    # Suppress performance warning since it is a byproduct of the simulation
    warnings.filterwarnings('ignore')

    # Get number of classes
    nb_classes = len(classes)

    # Performance metrics containers
    resampled_train_accuracy = pd.DataFrame(data=np.zeros(nb_bootstrap_samples).reshape(-1))
    resampled_train_precision = pd.DataFrame(data=np.zeros((nb_bootstrap_samples, nb_classes)), columns=classes)

    resampled_test_accuracy = resampled_train_accuracy.copy()
    resampled_test_precision = resampled_train_precision.copy()

    original_test_accuracy = resampled_train_accuracy.copy()
    original_test_precision = resampled_train_precision.copy()

    # Through each bootstrap sample
    for i in range(nb_bootstrap_samples):

        # Show iteration number
        print(i + 1)

        # Balance classes
        x_data_resampled, y_data_resampled = balance_classes(x_data, y_data)

        # Split into training and test sets
        x_train_resampled, x_test_resampled, y_train_resampled, y_test_resampled = \
            train_test_split(x_data_resampled, y_data_resampled, test_size=0.2)

        # Train and test the model on the resampled data
        y_train_pred_resampled, y_test_pred_resampled = \
            train_and_test_model(model, x_train_resampled, y_train_resampled, x_test_resampled)

        # Calculate training metrics on resampled data
        resampled_train_accuracy.iloc[i] = accuracy_score(y_train_resampled, y_train_pred_resampled)
        resampled_train_precision.iloc[i, :] = precision_score(y_train_resampled, y_train_pred_resampled, average=None)

        # Calculate test metrics on resampled data
        resampled_test_accuracy[i] = accuracy_score(y_test_resampled, y_test_pred_resampled)
        resampled_test_precision.iloc[i, :] = precision_score(y_test_resampled, y_test_pred_resampled, average=None)

        # Evaluate model on the original data
        y_pred_not_resampled = model.predict(x_data)

        # Calculate test metrics on original data
        original_test_accuracy[i] = accuracy_score(y_data, y_pred_not_resampled)
        original_test_precision.iloc[i, :] = precision_score(y_data, y_pred_not_resampled, average=None)

    print(f'\n\nRESAMPLED DATA {model_name}\n')

    train_acc_mean = resampled_train_accuracy.mean()[0]
    test_acc_mean = resampled_test_accuracy.mean()[0]

    train_acc_low, train_acc_up = bootstrap_calculate_confidence_interval(resampled_train_accuracy.values.flatten())
    test_acc_low, test_acc_up = bootstrap_calculate_confidence_interval(resampled_test_accuracy.values.flatten())

    train_prec_means = resampled_train_precision.mean().values
    test_prec_means = resampled_test_precision.mean().values

    train_prec_low = [0.0] * nb_classes
    train_prec_up = train_prec_low.copy()
    test_prec_low = train_prec_low.copy()
    test_prec_up = train_prec_low.copy()

    for i in range(nb_classes):
        train_prec_low[i], train_prec_up[i] = bootstrap_calculate_confidence_interval(resampled_train_precision.iloc[:, i].values.flatten())
        test_prec_low[i], test_prec_up[i] = bootstrap_calculate_confidence_interval(resampled_test_precision.iloc[:, i].values.flatten())

    print(f"\tTrain accuracy  : {train_acc_mean:.2f} [{train_acc_low:.2f}, {train_acc_up:.2f}]")
    print(f"\tTest  accuracy  : {test_acc_mean:.2f} [{test_acc_low:.2f}, {test_acc_up:.2f}]\n")

    for i, class_name in enumerate(classes):

        print(f"\tTrain precision {class_name}: {train_prec_means[i]:.2f} [{train_prec_low[i]:.2f}, {train_prec_up[i]:.2f}]")
        print(f"\tTest  precision {class_name}: {test_prec_means[i]:.2f} [{test_prec_low[i]:.2f}, {test_prec_up[i]:.2f}]\n")

    print(f'ORIGINAL DATA {model_name}\n')

    original_acc_mean = original_test_accuracy.mean().values[0]
    original_prec_means = original_test_precision.mean().values

    original_acc_low, original_acc_up = bootstrap_calculate_confidence_interval(original_test_accuracy.values.flatten())

    original_prec_low = [0.0] * nb_classes
    original_prec_up = original_prec_low.copy()

    for i in range(nb_classes):
        original_prec_low[i], original_prec_up[i] = bootstrap_calculate_confidence_interval(original_test_precision.iloc[:, i].values.flatten())

    print(f"\tTest  accuracy  : {original_acc_mean:.2f} [{original_acc_low:.2f}, {original_acc_up:.2f}]\n")

    for i, class_name in enumerate(classes):

        print(f"\tTest precision {class_name}: {original_prec_means[i]:.2f} [{original_prec_low[i]:.2f}, {original_prec_up[i]:.2f}]")

    # Clear warning filters
    warnings.resetwarnings()