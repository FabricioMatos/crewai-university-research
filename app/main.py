from datetime import datetime
import json
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

import agentops
from crewai import Crew, Process

from crew.agents import (
    reset_agents,
    university_finder,
    application_researcher_coordinator,
    application_analyst,
    cost_analyst,
    living_cost_analyst,
    relevance_analyst,
    quality_assurance_analyst,
)
from crew.models import UniversityList
from crew.tasks import (
    build_per_university_tasks,
    identify_universities_task,
    create_final_report_task,
)


def print_log(line: str):
    print()
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"========= [{current_time}] =========")
    print(f"{line}")
    print()


inputs_details = {
    "top_k_universities": 10,
    "master_program": "Quantum Computing",
    "region": "Europe",
}


def find_universities_candidates() -> UniversityList:
    universities_list: UniversityList = None

    # if the file were created in previous execution, return
    if os.path.exists("data/partial/UniversityList.json"):
        with open("data/partial/UniversityList.json", "r") as f:
            universities_list = UniversityList(**json.load(f))
    else:
        # If not, run the crew to find the universities
        print_log("Finding universities candidates...")
        university_research_crew = Crew(
            # embedder=embedder_config,
            agents=[university_finder()],
            tasks=[
                identify_universities_task,
            ],
            verbose=False,
            memory=True,
            process=Process.sequential,
        )

        # run the first crew to list the universities
        universities_list = university_research_crew.kickoff(
            inputs=inputs_details
        )

        # save for future re-executions
        with open("data/partial/UniversityList.json", "w") as f:
            json.dump(universities_list.dict(), f, indent=4)

    print_log(f"{len(universities_list.universities)} universities")
    print_log(f"universities_list: {universities_list}")

    return universities_list


def research_universities(universities_list):
    for university in universities_list.universities:
        reset_agents()
        tasks = build_per_university_tasks(university)

        # Create Crew to process new tasks/sub-tasks
        print_log(f"Researching university {university.name} ...")
        research_crew = Crew(
            # embedder=embedder_config,
            agents=[
                university_finder(),
                application_researcher_coordinator(),
                application_analyst(),
                cost_analyst(),
                living_cost_analyst(),
                relevance_analyst(),
                quality_assurance_analyst(),
            ],
            tasks=tasks,
            verbose=False,
            memory=True,
            cache=True,
            process=Process.sequential,
        )

        university_report = research_crew.kickoff(
            inputs={**inputs_details, **university.dict()}
        )
        print_log(f'\nuniversity "{university.name}":')
        print_log(f"usage_metrics: {research_crew.usage_metrics}")
        print_log(university_report)


def create_final_report(universities_list):
    # Create Crew to process new tasks/sub-tasks
    print_log("Creating final report...")
    final_report_crew = Crew(
        agents=[
            application_researcher_coordinator(),
            quality_assurance_analyst(),
        ],
        tasks=[
            create_final_report_task,
        ],
        verbose=False,
        memory=True,
        cache=True,
        process=Process.sequential,
    )

    final_report = final_report_crew.kickoff(
        inputs={**inputs_details, **universities_list.dict()}
    )
    print_log(f"final_report:\n\n{final_report}")


def main():
    # https://app.agentops.ai (dashboard)
    agentops.init(instrument_llm_calls=True)

    universities_list = find_universities_candidates()
    # research_universities(universities_list)
    create_final_report(universities_list)

    agentops.end_session("Success")


if __name__ == "__main__":
    main()
