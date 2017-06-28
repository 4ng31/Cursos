import numpy as np
import pandas as pd
import os


def classify(input_data, training_set, labels, k=1):
    """
    Uses kNN algorithm to classify input data given a set of
    known data.

    Args:
        input_data: Pandas Series of input data
        training_set: Pandas Data frame of training data
        labels: Pandas Series of classifications for training set
        k: Number of neighbors to use

    Returns:
        Predicted classification for given input data
    """
    distance_diff = training_set - input_data
    distance_squared = distance_diff**2
    distance = distance_squared.sum(axis=1)**0.5
    distance_df = pd.concat([distance, labels], axis=1)
    distance_df.sort(columns=[0], inplace=True)
    top_knn = distance_df[:k]
    return top_knn[1].value_counts().index.values[0]


def load_data(directory):
    """
    Loads text files of digits in directory as list of lists,
    where each row is represents the digit in a series 0's and 1's

    Each digit is 32 x 32.

    Args:
        directory: Directory that contains text files of digits

    Returns:
        List of lists containing 0's and 1's representing each digit
    """
    dataset = []
    labels = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath) as infile:
            vector = []
            for line in infile:
                vector.extend(line.strip())
            dataset.append(vector)
        labels.append(int(filename[0]))
    return dataset, labels


def main():

    # Load data
    raw_training_data, raw_training_labels = load_data('../sample/Ch02/trainingDigits/')
    raw_test_data, raw_test_labels = load_data('../sample/Ch02/testDigits/')

    # Convert data into Pandas data structures
    training_labels = pd.Series(raw_training_labels)
    training_data = pd.DataFrame.from_records(np.array(raw_training_data, int))

    test_labels = pd.Series(raw_test_labels)
    test_data = pd.DataFrame.from_records(np.array(raw_test_data, int))

    # Apply kNN algorithm to all test data
    result_df = test_data.apply(lambda row: classify(row, training_data, training_labels, k=3), axis=1)

    # Calculate the number of correct predictions
    error_df = result_df == test_labels
    print error_df.value_counts()


if __name__ == '__main__':
    main()
