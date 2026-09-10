from src.utils import parse_bool, safe_json


def test_parse_bool():
    assert parse_bool("true")
    assert not parse_bool("false")


def test_safe_json():
    assert safe_json('{"x": 1}')["x"] == 1
