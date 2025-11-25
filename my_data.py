from os.path import join

from my_dataset_2 import DatasetFromFolder

def get_training_set(root_dir, direction):
    """Perform the get_training_set operation.

    Args:
        root_dir (str): Description.
        direction (str): Description.

    Returns:
        Any: Result.
    """
    # train_dir = join(root_dir, "train")

    return DatasetFromFolder(root_dir, direction)

def get_test_set(root_dir, direction):
    """Perform the get_test_set operation.

    Args:
        root_dir (str): Description.
        direction (str): Description.

    Returns:
        Any: Result.
    """
    # test_dir = join(root_dir, "test")

    return DatasetFromFolder(root_dir, direction)
