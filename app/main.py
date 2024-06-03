import json
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# https://app.agentops.ai/start
import agentops

agentops.init()

from crewai import Crew, Process

from crew.agents import university_finder
from crew.models import University, UniversityList
from crew.utils import str_to_model
from crew.tasks import identify_universities_task

import logging

# Suppress CrewAI logs
logging.getLogger("crewai").setLevel(logging.CRITICAL)
logging.getLogger("crewai-tools").setLevel(logging.CRITICAL)


def find_universities_candidates() -> UniversityList:
    # if the file were created in previous execution, return
    if os.path.exists("data/partial/UniversityList.json"):
        with open("data/partial/UniversityList.json", "r") as f:
            return UniversityList(**json.load(f))

    # If not, run the crew to find the universities
    university_research_crew = Crew(
        agents=[university_finder],
        tasks=[
            identify_universities_task,
        ],
        verbose=False,
        memory=True,
        process=Process.sequential,
    )

    inputs_details = {
        "top_k_universities": 100,
        "region": "Europe",
    }

    # run the first crew to list the universities
    universities_list = university_research_crew.kickoff(inputs=inputs_details)
    with open('data/partial/UniversityList.json', 'w') as f:
        json.dump(universities_list.dict(), f, indent=4)

    return universities_list


def main():
    universities_list = find_universities_candidates()
    print(len(universities_list.universities), "universities")
    print(universities_list.universities[0])

    countries = set([u.country for u in universities_list.universities])
    print(len(countries), "countries:")
    print(countries)

    agentops.end_session('Success')


main()
