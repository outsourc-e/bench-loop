import pytest
from src import parse_row, parse_csv, filter_active


def test_basic_parse():
    row = {"name": "Alice", "age": "30", "salary": "75000.50", "active": "true"}
    result = parse_row(row)
    assert result == {"name": "Alice", "age": 30, "salary": 75000.50, "active": True}


def test_bool_variants():
    assert parse_row({"name": "A", "age": "1", "salary": "1", "active": "YES"})["active"] is True
    assert parse_row({"name": "A", "age": "1", "salary": "1", "active": "0"})["active"] is False
    assert parse_row({"name": "A", "age": "1", "salary": "1", "active": "False"})["active"] is False


def test_missing_field_is_none():
    result = parse_row({"name": "Bob"})
    assert result["age"] is None
    assert result["salary"] is None
    assert result["active"] is None


def test_empty_string_is_none():
    result = parse_row({"name": "", "age": "", "salary": "", "active": ""})
    assert result["name"] is None


def test_extra_fields_dropped():
    result = parse_row({"name": "Carol", "age": "25", "salary": "50000", "active": "no", "department": "eng"})
    assert "department" not in result


def test_invalid_int_raises():
    with pytest.raises(ValueError, match="age"):
        parse_row({"name": "X", "age": "not_a_number", "salary": "1", "active": "true"})


def test_filter_active_basic():
    rows = parse_csv([
        {"name": "A", "age": "20", "salary": "50000", "active": "true"},
        {"name": "B", "age": "30", "salary": "60000", "active": "false"},
    ])
    active = filter_active(rows)
    assert len(active) == 1
    assert active[0]["name"] == "A"


def test_filter_active_with_none():
    """Rows where active is None should NOT be included in active list."""
    rows = parse_csv([
        {"name": "A", "age": "20", "salary": "50000", "active": "yes"},
        {"name": "B", "age": "30", "salary": "60000"},  # active will be None
        {"name": "C", "age": "25", "salary": "70000", "active": "no"},
    ])
    active = filter_active(rows)
    assert len(active) == 1
    assert active[0]["name"] == "A"
