"""Offline tests: they only exercise Pydantic validation, so no API calls are made."""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas import Extraction  # noqa: E402


def test_valid_json_is_parsed():
    data = '{"summary": "ok", "tasks": [{"description": "Send report", "due_date": "2026-11-20", "priority": "high"}]}'
    result = Extraction.model_validate_json(data)
    assert result.tasks[0].due_date.day == 20


def test_invalid_date_is_rejected():
    with pytest.raises(ValidationError):
        Extraction.model_validate_json('{"summary": "x", "dates": [{"date": "2026-02-30", "description": "?"}]}')


def test_unknown_priority_is_rejected():
    with pytest.raises(ValidationError):
        Extraction.model_validate_json('{"summary": "x", "tasks": [{"description": "a", "priority": "urgent"}]}')


def test_short_name_is_rejected():
    with pytest.raises(ValidationError):
        Extraction.model_validate_json('{"summary": "x", "people": [{"name": " A "}]}')


def test_lists_default_to_empty():
    result = Extraction.model_validate_json('{"summary": "nothing"}')
    assert result.people == [] and result.dates == [] and result.tasks == []