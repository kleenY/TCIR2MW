"""Dataset path definitions and lightweight helpers consumed by dataset loaders and scripts. Representative functions: get_training_set, get_test_set."""

from os.path import join

from my_dataset_2 import DatasetFromFolder


# Purpose: Core routine of this module; see code for tensor shapes and exact semantics.
def get_training_set(root_dir, direction):
    # train_dir = join(root_dir, "train")

    return DatasetFromFolder(root_dir, direction)


def get_test_set(root_dir, direction):
    # test_dir = join(root_dir, "test")

    return DatasetFromFolder(root_dir, direction)