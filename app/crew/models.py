from typing import List, Literal
from pydantic import BaseModel


class University(BaseModel):
    name: str
    country: str
    city: str


class UniversityList(BaseModel):
    universities: List[University]


class UniversityFee(BaseModel):
    anual_fee_usd: float
    fee_details: str


class MinMax(BaseModel):
    min: float
    max: float


class UniversityLivingCosts(BaseModel):
    anual_living_cost_usd: MinMax
    living_costs_details: MinMax


class UniversityRelevance(BaseModel):
    level: Literal['low', 'medium', 'high']
    main_researchers_list: List[str]
    main_research_fields_list: List[str]


class UniversityApplicatnsTestimonies(BaseModel):
    success_testimonies_list: List[str]
    failure_testimonies_list: List[str]


class UniversityDetails(BaseModel):
    university: University
    fee: UniversityFee
    living_costs: UniversityLivingCosts
    relevance: UniversityRelevance
    testimonies: UniversityApplicatnsTestimonies
