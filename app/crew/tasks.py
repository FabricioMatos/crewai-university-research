import logging
import re
from typing import List, Type
from pydantic import BaseModel
from crewai import Task

from crew.tools import (
    search_tool,
    scrape_tool,
    read_partial_reports_tool,
)
from crew.agents import (
    acceptance_criteria,
    university_finder,
    application_researcher_coordinator,
    application_analyst,
    cost_analyst,
    living_cost_analyst,
    relevance_analyst,
)
from crew.models import (
    University,
    UniversityApplicatnsTestimonies,
    UniversityDetails,
    UniversityFee,
    UniversityList,
    UniversityLivingCosts,
    UniversityRelevance,
)

logger = logging.getLogger("tasks")

COMMON_TOOLS = [
    search_tool,
    scrape_tool,
]


def get_model_schema_instructions(model_class: Type[BaseModel]) -> str:
    schema = f"{model_class.schema_json(indent=2)}"
    schema = schema.replace("{", "{{").replace("}", "}}")
    logger.info(f"{model_class.__name__}: {schema}")
    return (
        "\nThe response should be a JSON object with the following schema:\n\n"
        f"{schema}"
    )


# Task 1: Identify Top Universities
identify_universities_task = Task(
    description=(
        "List the top {top_k_universities} "
        "universities in {region} offering master programs "
        "with a focus on {master_program}. Confirm that ALL "
        "universities are in {region}!"
    ),
    expected_output=(
        "A list of {top_k_universities} universities in {region} "
        "offering top-tier master programs in {master_program}. "
        " Get its name, country and city in a JSON format with "
        "(universitie's) name, city and country. The final list should "
        "have at least 50 universities."
        f"{get_model_schema_instructions(UniversityList)}"
    ),
    agent=university_finder(),
    async_execution=False,
    tools=COMMON_TOOLS,
    output_pydantic=UniversityList,
    output_file="data/partial/identify_universities_task.txt",
)


def build_per_university_tasks(university: University) -> List[Task]:
    tasks = []
    filename_prefix = f"{university.city}-{university.country}"
    filename_prefix = re.sub(r"[^a-zA-Z0-9-_]", "", filename_prefix)

    # Sub-Task 2.1: Analyze University Fees
    analyze_fees_task = Task(
        description=(
            "Analyze the university fees involved for a master program in "
            f"{{master_program}} at {university.city}/{university.country} "
            "university and check for any incentives available for "
            "Brazilian/foreign applicants."
        ),
        expected_output=(
            f"The anual {university.city}/{university.country} fee "
            "in USD and the detailed breakdown of university fees in "
            "the `fee_details` property, with the main cost components "
            "and info about any available financial incentives with "
            "particular attention to specific details that may matter "
            "for Brazilian applicants, in a JSON format: anual_fee_usd: "
            "15000.00, fee_details: details ...."
            f"{get_model_schema_instructions(UniversityFee)}"
        ),
        agent=cost_analyst(),
        async_execution=False,
        tools=COMMON_TOOLS,
        output_pydantic=UniversityFee,
        output_file=f"data/partial/{filename_prefix}_fees.txt",
    )
    tasks.append(analyze_fees_task)

    # Sub-Task 2.2: Analyze Living Costs
    analyze_living_cost_task = Task(
        description=(
            f"Analyze living costs in {university.city}/{university.country} "
            f"near {university.name}, including housing, food, leisure, "
            "and healthcare. Pay attention to and flag any restrictions "
            "for applicants from Brazil."
        ),
        expected_output=(
            "The anual living costs in USD at "
            f"{university.city}/{university.country} in a JSON format "
            "like in: 'anual_living_cost_usd': 15000.00, "
            "living_costs_details: details .... "
            "The detailed breakdown of living costs fees should cover: "
            "housing, food, health, pleasure, etc."
            f"{get_model_schema_instructions(UniversityLivingCosts)}"
        ),
        agent=living_cost_analyst(),
        async_execution=False,
        tools=COMMON_TOOLS,
        output_pydantic=UniversityLivingCosts,
        output_file=f"data/partial/{filename_prefix}_living_costs.txt",
    )
    tasks.append(analyze_living_cost_task)

    # Task 2.3: Analyze Relevance
    analyze_relevance_task = Task(
        description=(
            "Analyze the relevance of the university "
            f"{university.city}/{university.country} in "
            "{master_program} by examining the main researchers "
            "in the field and the main topics in the field with "
            "relevant articles published in the last 5 years."
        ),
        expected_output=(
            f"An analysis of the {university.city}/{university.country} "
            "university's relevance in {master_program} based "
            "on researchers and publications. Fill the list of "
            "main top 3 researchers and research topics in the "
            "`main_researchers` and `main_research_field` fields "
            "respectively. Fill the `level` field according your assessment."
            f"{get_model_schema_instructions(UniversityRelevance)}"
        ),
        agent=relevance_analyst(),
        async_execution=False,
        tools=COMMON_TOOLS,
        output_pydantic=UniversityRelevance,
        output_file=f"data/partial/{filename_prefix}_relevance.txt",
    )
    tasks.append(analyze_relevance_task)

    # Task 2.4: Add testimonies
    add_testimonies_task = Task(
        description=(
            "Retrive testimonies from other past applicants to "
            f"{university.city}/{university.country} that succeed "
            "and also failed. Search for testimonies of both success "
            "and failures of similar applicants worldwide, "
            "especially those from South America and Brazil. "
        ),
        expected_output=(
            f"Each testimony of {university.city}/{university.country} "
            "applicants should be summarized with its most inspiring and "
            "informative highlights. Always add a link with reference "
            "to the source of the testimony. "
            f"{get_model_schema_instructions(UniversityApplicatnsTestimonies)}"
        ),
        agent=application_analyst(),
        async_execution=False,
        tools=COMMON_TOOLS,
        output_pydantic=UniversityApplicatnsTestimonies,
        output_file=f"data/partial/{filename_prefix}_testimonies.txt",
    )
    tasks.append(add_testimonies_task)

    # Task 3: Analyze Application Requirements
    analyze_application_task = Task(
        description=(
            "Review and aggregate all information gathered about "
            f"{university.city}/{university.country} university "
            "for a application in one JSON object, combining fees, "
            "living costs, relevance in {master_program} and testimonies."
        ),
        expected_output=(
            "Combine the information of the university "
            f"{university.city}/{university.country} and the details about "
            "university fees, living costs, and relevance in the field in "
            "one object that aggregates all properties in one JSON with the "
            "same property names. And make sure to comply with the following "
            "acceptance criteria:\n "
            f"{acceptance_criteria}"
            f"{get_model_schema_instructions(UniversityDetails)}"
        ),
        agent=application_researcher_coordinator(),
        async_execution=False,
        tools=COMMON_TOOLS,
        context=tasks,
        output_pydantic=UniversityDetails,
        output_file=f"data/partial/{filename_prefix}_all_details.txt",
    )
    tasks.append(analyze_application_task)

    # Task 4: Build a final report in markdown format for this university
    build_partial_report_task = Task(
        description=(
            "Aggregate the collected information by the other team members "
            f"about {university.city}/{university.country} , and write a "
            "markdown comprehensive report with all relevant information "
            "collected by the other team members."
        ),
        expected_output=(
            "A final Markdown report about "
            f"{university.city}/{university.country} covering "
            "ALL ITEMS mentioned in the acceptance criteria:\n "
            f"{acceptance_criteria} "
            f"{get_model_schema_instructions(UniversityDetails)}"
        ),
        agent=application_researcher_coordinator(),
        async_execution=False,
        tools=COMMON_TOOLS,
        context=[analyze_application_task],
        output_file=f"data/partial_report_{filename_prefix}.md",
    )
    tasks.append(build_partial_report_task)

    return tasks


# Task 5: Create the finall report
create_final_report_task = Task(
    description=(
        "List and aggregate all partial reports available as "
        "markdown files (*.md) in the local `data` folder using "
        " LocalFilesSearchTool and FileReadTool accordingly. "
        "There is one file for each university researched. "
        "At the end, create a final report with all the information "
        "aggregated in markdown as defined in the acceptance criteria."
    ),
    expected_output=(
        "A final Markdown report aggregating all partial reports, "
        " covering ALL ITEMS mentioned in the acceptance criteria:\n "
        f"{acceptance_criteria} "
    ),
    agent=application_researcher_coordinator(),
    async_execution=False,
    # avoid using common tools here to avoid confusion
    tools=[read_partial_reports_tool],
    output_file="data/final_report.md",
)
