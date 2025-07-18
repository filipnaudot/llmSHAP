import pytest
from llmSHAP.data_handler import DataHandler

def test_initialization():
    sample_data = {"a": "data_1", "b": 2222, "c": "data_3"}
    expected_enum = {0: "a", 1: "b", 2: "c"}

    handler = DataHandler(sample_data)

    assert handler.data == sample_data
    assert handler.key_enum == expected_enum

def test_empty_data():
    sample_data = {}
    handler = DataHandler(sample_data)
    assert handler.data == sample_data
    assert handler.key_enum == {}

def test_remove_single_no_mask():
    sample = {"a": 0, "b": 1, "c": 2, "d": 3}
    handler = DataHandler(sample)
    new_data = handler.remove(1, mask=False)
    expected = {"a": 0, "c": 2, "d": 3}
    
    assert new_data == expected
    assert handler.data == sample

def test_remove_single_mask():
    sample = {"a": 0, "b": 1, "c": 2, "d": 3}
    handler = DataHandler(sample)
    new_data = handler.remove(1, mask=True)
    expected = {"a": 0, "b": "", "c": 2, "d": 3}
    
    assert new_data == expected
    assert handler.data == sample

def test_remove_multiple_no_mask():
    sample = {"a": 0, "b": 1, "c": 2, "d": 3}
    handler = DataHandler(sample)
    new_data = handler.remove({1, 2}, mask=False)
    expected = {"a": 0, "d": 3}
    
    assert new_data == expected
    assert handler.data == sample

def test_remove_multiple_mask():
    sample = {"a": 0, "b": 1, "c": 2, "d": 3}
    handler = DataHandler(sample)
    new_data = handler.remove({1, 2}, mask=True)
    expected = {"a": 0, "b": "", "c": "", "d": 3}
    
    assert new_data == expected
    assert handler.data == sample

def test_get_data_mask():
    sample = {"a": 0, "b": 1, "c": 2, "d": 3}
    handler = DataHandler(sample)
    data = handler.get_data(2, mask=True)
    expected = {"a": "", "b": "", "c": 2, "d": ""}
    
    assert data == expected
    assert handler.data == sample

def test_get_data_no_mask():
    sample = {"a": 0, "b": 1, "c": 2, "d": 3}
    handler = DataHandler(sample)
    data = handler.get_data(2, mask=False)
    expected = {"c": 2}
    
    assert data == expected
    assert handler.data == sample

def test_permanent_keys():
    sample = {"a": 0, "b": 1, "c": 2, "d": 3}
    handler = DataHandler(sample, permanent_key_names={"a", "d"})
    data = handler.get_data(2, mask=False)
    expected = {"a": 0, "c": 2, "d": 3}
    
    assert data == expected
    assert handler.data == sample

def test_get_keys_include_permanent():
    sample = {"a": 5, "b": 5, "c": 5, "d": 5}
    permanent_keys = {"b", "c"}
    handler = DataHandler(sample, permanent_key_names=permanent_keys)
    keys = handler.get_keys()
    expected = [0,1,2,3]
    
    assert keys == expected
    assert handler.data == sample

def test_get_keys_exclude_permanent():
    sample = {"a": 5, "b": 5, "c": 5, "d": 5}
    permanent_keys = {"b", "c"}
    handler = DataHandler(sample, permanent_key_names=permanent_keys)
    keys = handler.get_keys(exclude_permanent_keys=True)
    expected = [0,3]
    
    assert keys == expected
    assert handler.data == sample

def test_to_string():
    sample = {"a": "This", "b": "is", "c": "a", "d": "test."}
    handler = DataHandler(sample)
    data_string = handler.to_string()
    
    assert data_string == "This is a test."

def test_string_input():
    sample = "This is a string."
    handler = DataHandler(sample)

    keys = handler.get_keys()
    assert keys == [0,1,2,3]

    data_string = handler.to_string()
    assert data_string == "This is a string."