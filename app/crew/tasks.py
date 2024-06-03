from typing import List
from pydantic import BaseModel
from crewai import Task

from crew.tools import (
    search_tool,
    scrape_tool,
    local_llm_tool,
)
from crew.agents import (
    university_finder,
)
"""
    application_analyst,
    application_analyst,
    cost_analyst,
    living_cost_analyst,
    quantum_relevance_analyst,
    quality_assurance_analyst,
"""
from crew.models import UniversityList
# from agents import acceptance_criteria


# Task 1: Identify Top Universities
identify_universities_task = Task(
    description=(
        "List the top {top_k_universities} non-UK European "
        "universities offering theoretical physics programs "
        "with a focus on quantum physics. Confirm that ALL "
        "universities are in EUROPE and not in the United Kingdom."
    ),
    expected_output=(
        "A list of {top_k_universities} non-UK European universities "
        "offering top-tier programs in theoretical physics. "
        " Get its name, country and city in a JSON format with "
        "(universitie's) name, city and country. The final list should "
        "have at least 50 universities."
    ),
    agent=university_finder,
    async_execution=False,
    tools=[local_llm_tool, scrape_tool, search_tool],
    output_pydantic=UniversityList,
    output_file="data/partial/identify_universities_task.txt",
)
