from datetime import datetime
from crewai import Agent
from crew.tools import (
    search_tool,
    scrape_tool,
)

agents = {}


def reset_agents():
    global agents
    agents = {}


acceptance_criteria = (
    "1. The report 2 sections: '1. Summary' and '2. Details'.\n "
    "2. The Summary section have a table with the top universities "
    "selected ranked by total costs in descending order. "
    "Also present in the summary table the main research topics "
    "of each university .\n "
    "3. The Details section have detailed information about each "
    "of the top universities, including:\n "
    "  - breaking down of main compomentes of cost of living.\n "
    "  - univerisity fees.\n "
    "  - application details.\n "
    "  - acceptance rate.\n ")


def step_callback(next_step_output: str):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f'[{current_time}]\n {next_step_output}')


# Agent 1: University Finder
def university_finder() -> Agent:
    if "university_finder" not in agents:
        agents['university_finder'] = Agent(
            # llm=llm_agents,
            role="University Finder",
            goal=(
                "Identify and list the top universities in {region} offering "
                "{master_program} programs."
            ),
            backstory=(
                "As a University Finder, your expertise in searching and "
                "evaluating academic institutions ensures that only the "
                "top universities for {master_program} are listed."
            ),
            tools=[scrape_tool, search_tool],
            allow_delegation=False,
            verbose=False,
            cache=True,
            step_callback=step_callback,
        )
    return agents["university_finder"]


# Agent 2: Application Researcher Coordinator
def application_researcher_coordinator() -> Agent:
    if "application_researcher_coordinator" not in agents:
        agents['application_researcher_coordinator'] = Agent(
            # llm=llm_agents,
            role="Application Researcher Coordinator",
            goal=(
                "Analyse an university from many different "
                "aspects (fees, living costs, relevance, etc) "
                "and write a report comparing them."
            ),
            backstory=(
                "As an Application Researcher Coordinator, you specialize "
                "in coordinate the team to analyze universities and "
                "produce a comprehensive report comparing all of them."
            ),
            tools=[scrape_tool, search_tool],
            allow_delegation=True,
            verbose=False,
            cache=True,
            step_callback=step_callback,
        )
    return agents["application_researcher_coordinator"]


# Agent 3: Application Analyst
def application_analyst() -> Agent:
    if "application_analyst" not in agents:
        agents['application_analyst'] = Agent(
            # llm=llm_agents,
            role="Application Analyst",
            goal=(
                "Analyze and summarize all the information collected from any "
                "specific university."),
            backstory=(
                "As an Application Analyst, you specialize in dissecting the "
                "application processes of universities, making sure every "
                "requirement and deadline is clearly understood."
            ),
            tools=[scrape_tool, search_tool],
            allow_delegation=True,
            verbose=False,
            cache=True,
            step_callback=step_callback,
        )
    return agents["application_analyst"]


# Agent 4: Cost Analyst
def cost_analyst() -> Agent:
    if "cost_analyst" not in agents:
        agents['cost_analyst'] = Agent(
            # llm=llm_agents,
            role="University Cost Analyst",
            goal=(
                "Analyze university fees and any available incentives for "
                "Brazilian applicants in USD."
            ),
            backstory=(
                "As a Cost Analyst, your ability to break down university fees "
                "and identify financial aids or incentives helps ensure that "
                "education remains affordable."
            ),
            tools=[scrape_tool, search_tool],
            allow_delegation=False,
            verbose=False,
            cache=True,
            step_callback=step_callback,
        )
    return agents["cost_analyst"]


# Agent 5: Living Cost Analyst
def living_cost_analyst() -> Agent:
    if "living_cost_analyst" not in agents:
        agents['living_cost_analyst'] = Agent(
            # llm=llm_agents,
            role="Living Cost Analyst",
            goal=(
                "Analyze living costs in the cities where the universities "
                "are located."
            ),
            backstory=(
                "As a Living Cost Analyst, you provide detailed insights into "
                "the cost of living in various cities, covering all necessary "
                "aspects from housing to healthcare."
            ),
            tools=[scrape_tool, search_tool],
            allow_delegation=False,
            verbose=False,
            cache=True,
            step_callback=step_callback,
        )
    return agents["living_cost_analyst"]


# Agent 6: Relevance Analyst
def relevance_analyst() -> Agent:
    if "relevance_analyst" not in agents:
        agents['relevance_analyst'] = Agent(
            # llm=llm_agents,
            role="{master_program} Relevance Analyst",
            goal=(
                "Analyze the relevance of the universities in the field of "
                "{master_program} by examining researchers and "
                "published articles."
            ),
            backstory=(
                "As a {master_program} Relevance Analyst, your deep dive "
                "into the academic contributions and researchers at "
                "universities ensures that the institutions chosen "
                "are leaders in the field."
            ),
            tools=[scrape_tool, search_tool],
            allow_delegation=False,
            verbose=False,
            cache=True,
            step_callback=step_callback,
        )
    return agents["relevance_analyst"]


# Agent 7: QA
def quality_assurance_analyst() -> Agent:
    if "quality_assurance_analyst" not in agents:
        agents['quality_assurance_analyst'] = Agent(
            # llm=llm_agents,
            role="Research Quality Assurance Specialist",
            goal=(
                "Analyze the final report and check if it meets the following "
                f"acceptance criteria:\n {acceptance_criteria}"),
            backstory=(
                "As a Research Quality Assurance Specialist, you ensure that "
                "the final report meets all specified criteria, providing a "
                "comprehensive and reliable overview of the universities."
            ),
            tools=[scrape_tool, search_tool],
            allow_delegation=True,
            verbose=False,
            cache=True,
            step_callback=step_callback,
        )
    return agents["quality_assurance_analyst"]
