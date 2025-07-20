from itertools import chain, combinations


# Credit: https://github.com/TimKam/Quantitative-Bipolar-Argumentation/blob/main/qbaf_ctrbs/utils.py
def determine_powerset(elements):
    elements_list = list(elements)
    powerset_elements = chain.from_iterable(combinations(elements_list, option) for option in range(len(elements_list) + 1))
    return [set(powerset_element) for powerset_element in powerset_elements]