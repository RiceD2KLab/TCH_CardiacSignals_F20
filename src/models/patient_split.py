"""
For a given patient, split the data into the first third and remaining two thirds. First third is considered to be
normal heartbeats. Randomly select half of the heartbeats of the first third to train on and other half for testing.
"""
import numpy as np

def patient_split_train(filepath, split_ratio):
    """
    patient_split splits file by the split ratio, first portion is split into train and test of "normal" heartbeats,
    and last portion is determined to be abnormal
    :param filepath: [str] filepath of file to be split as string
    :param split_ratio: [float] value from 0 to 1 for the data split
    :return: training, testing, and irregular heartbeat sets as .npy arrays
    """
    data = np.load(filepath)
    splitting_idx = round(len(data) * split_ratio)
    first_portion = data[0:splitting_idx]
    second_portion = data[splitting_idx:]
    np.random.shuffle(first_portion)
    half_idx = round(first_portion.shape[0] / 2)
    half1, half2 = first_portion[:half_idx, :], first_portion[half_idx:, :]

    return half1, half2, second_portion


def patient_split_all(filepath, split_ratio):
    """
    patient_split splits file by the split ratio, the first portion is considered "normal" heartbeats
    and the last section is abnormal, returns full array for encoding as well
    other two thirds is considered abnormal data
    :param filepath: [str] filepath of file to be split as string
    :param split_ratio: [float] float value from 0 to 1 for the data split
    :return: normal and irregular heartbeat sets as .npy arrays
    """
    data = np.load(filepath)
    splitting_idx = round(len(data) * split_ratio)
    first_portion = data[0:splitting_idx]
    second_portion = data[splitting_idx:]

    return first_portion, second_portion, data
