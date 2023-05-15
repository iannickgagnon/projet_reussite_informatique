
# External libraries
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def train_and_test_model(model, x_train, y_train, x_test):

    # Train model
    model.fit(x_train, y_train)

    # Make predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    return y_train_pred, y_test_pred


def calculate_performance_metrics(y_train, y_train_pred, y_test, y_test_pred, labels=None, verbose=True):

    # Calculate training performance metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average=None)
    train_recall = recall_score(y_train, y_train_pred, average=None)
    train_f1 = f1_score(y_train, y_train_pred, average=None)

    # Calculate test performance metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average=None)
    test_recall = recall_score(y_test, y_test_pred, average=None)
    test_f1 = f1_score(y_test, y_test_pred, average=None)

    # Show performance metrics
    if verbose:
        print("\tTraining Accuracy  :", train_accuracy)
        print("\tTest Accuracy      :", test_accuracy)
        print("\n\tTraining Precision :", ' '.join([f'{label}: {precision:.2f}\t' for label, precision in zip(labels, train_precision)]))
        print("\tTest Precision     :", ' '.join([f'{label}: {precision:.2f}\t' for label, precision in zip(labels, test_precision)]))
        print("\n\tTraining Recall    :", ' '.join([f'{label}: {recall:.2f}\t' for label, recall in zip(labels, train_recall)]))
        print("\tTest Recall        :", ' '.join([f'{label}: {recall:.2f}\t' for label, recall in zip(labels, test_recall)]))
        print("\n\tTraining F1 Score  :", ' '.join([f'{label}: {f1:.2f}\t' for label, f1 in zip(labels, train_f1)]))
        print("\tTest F1 Score      :", ' '.join([f'{label}: {f1:.2f}\t' for label, f1 in zip(labels, train_f1)]))