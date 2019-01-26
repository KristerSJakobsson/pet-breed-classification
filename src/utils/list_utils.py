from typing import List, TypeVar

T = TypeVar('T')


def extract_unique_values(non_unique_list: List[T]) -> List[T]:
    """
    Gets unique values from list
    :param non_unique_list: A list with repeating items
    :return: A list without repeating items
    """
    return list(set(non_unique_list))
