"""Pydantic models that define the extraction contract.

The same models are used to (1) tell the LLM which JSON shape to produce
and (2) validate the response that comes back.
"""
import datetime as dt
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Person(BaseModel):
    name: str = Field(description="Person's name exactly as it appears in the text")
    email: Optional[str] = Field(default=None, description="Email address, if mentioned")
    role: Optional[str] = Field(default=None, description="Job title or role, if mentioned")

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("name is too short")
        return value


class KeyDate(BaseModel):
    date: dt.date = Field(description="Date in YYYY-MM-DD format")
    description: str = Field(description="What happens on this date")


class Task(BaseModel):
    description: str = Field(description="Action to be done, starting with a verb")
    owner: Optional[str] = Field(default=None, description="Who is responsible for the task")
    due_date: Optional[dt.date] = Field(default=None, description="Deadline in YYYY-MM-DD, if any")
    priority: Literal["high", "medium", "low"] = Field(
        default="medium", description="Inferred from the tone/urgency of the text"
    )


class Extraction(BaseModel):
    summary: str = Field(description="One-sentence summary of the text")
    people: list[Person] = Field(default_factory=list)
    dates: list[KeyDate] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)