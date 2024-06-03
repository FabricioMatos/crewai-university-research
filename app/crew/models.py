from typing import List
from pydantic import BaseModel


class University(BaseModel):
    name: str
    country: str
    city: str


class UniversityList(BaseModel):
    universities: List[University]
