from crewai import Agent
from crew.tools import (
    search_tool,
    scrape_tool,
    local_llm_tool,
)


acceptance_criteria = (
    "1. The report 2 sections: '1. Summary' and '2. Details'.\n "
    "2. The Summary section have a table with the top universities "
    "selected ranked by total costs in descending order.\n "
    "3. The Details section have detailed information about each "
    "of the top universities, including:\n "
    "  - breaking down of main compomentes of cost of living.\n "
    "  - univerisity fees.\n "
    "  - application details.\n "
    "  - acceptance rate.\n ")

# Agent 1: University Finder
university_finder = Agent(
    role="University Finder",
    goal=(
        "Identify and list the top universities in Europe offering "
        "theoretical physics programs with a focus on "
        "quantum physics."
    ),
    backstory=(
        "As a University Finder, your expertise in searching and "
        "evaluating academic institutions ensures that only the "
        "top universities for theoretical physics are listed."
    ),
    tools=[local_llm_tool, scrape_tool, search_tool],
    allow_delegation=False,
    verbose=False,
    cache=True,
)
"""
# Agent 2: Application Researcher
application_analyst = Agent(
    role="Application Researcher",
    goal=(
        "Analyse and compare European universisies from many different "
        "aspects (fees, living costs, relevance, etc) "
        "and write a report comparing them."
    ),
    tools=[scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As an Application Researcher, you specialize in coordinate the "
        "team to analyze universities and produce a comprehensive report "
        "comparing all of them."
    ),
    allow_delegation=True,
)

# Agent 3: Application Analyst
application_analyst = Agent(
    role="Application Analyst",
    goal=(
        "Analyze and summarize all the information collected from any "
        "specific university."),
    tools=[scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As an Application Analyst, you specialize in dissecting the "
        "application processes of universities, making sure every "
        "requirement and deadline is clearly understood."
    ),
    allow_delegation=False,
)

# Agent 4: Cost Analyst
cost_analyst = Agent(
    role="University Cost Analyst",
    goal=(
        "Analyze university fees and any available incentives for "
        "Brazilian applicants in USD."
    ),
    tools=[scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As a Cost Analyst, your ability to break down university fees "
        "and identify financial aids or incentives helps ensure that "
        "education remains affordable."
    ),
    allow_delegation=False,
)

# Agent 5: Living Cost Analyst
living_cost_analyst = Agent(
    role="Living Cost Analyst",
    goal=(
        "Analyze living costs in the cities where the universities "
        "are located."
    ),
    tools=[scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As a Living Cost Analyst, you provide detailed insights into "
        "the cost of living in various cities, covering all necessary "
        "aspects from housing to healthcare."
    ),
    allow_delegation=False,
)

# Agent 6: Quantum Physics Relevance Analyst
quantum_relevance_analyst = Agent(
    role="Quantum Physics Relevance Analyst",
    goal=(
        "Analyze the relevance of the universities in the field of "
        "quantum physics by examining researchers and published articles."
    ),
    tools=[scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As a Quantum Physics Relevance Analyst, your deep dive into the "
        "academic contributions and researchers at universities ensures "
        "that the institutions chosen are leaders in the field."
    ),
    allow_delegation=False,
)

# Agent 7: Research Quality Assurance Specialist
quality_assurance_analyst = Agent(
    role="Research Quality Assurance Specialist",
    goal=(
        "Analyze the final report and check if it meets the following "
        f"acceptance criteria:\n {acceptance_criteria}"),
    tools=[scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As a Research Quality Assurance Specialist, you ensure that "
        "the final report meets all specified criteria, providing a "
        "comprehensive and reliable overview of the universities."
    ),
    allow_delegation=False,
)
"""
