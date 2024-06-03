# to log the results while seeing them in the screen:
# python -u bruno-universities-ollama.py | tee logs_ollama_single_crew.txt

# Warning control
import warnings
warnings.filterwarnings('ignore')

import os
import json 
import re
import requests
from dotenv import load_dotenv

from pydantic import BaseModel
from typing import Type, TypeVar, List, Union, Literal

from crewai import Agent, Task, Crew, Process
from crewai_tools import (
  ScrapeWebsiteTool,
  SerperDevTool,
  BaseTool
)

import logging 
logger_crewai = logging.getLogger('crewai')

load_dotenv()

for key in ['SERPER_API_KEY']:
    assert os.environ.get(key, '') != '', f'{key} env var missing!'


OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
VERSION = "v2_llama3"

class SkipOrOp:
    def run(self, *args, **kwargs):
        return None

    def __or__(self, other):
        return other


class LocalLLMTool(BaseTool):
    name: str = "[Default tool] Ollama Local LLM chat tool"
    description: str = "A tool to interact with a local LLM served by Ollama. Expets a string as input."
    
    def _run(self, argument: str) -> str:
        def truncate_argument(argument: str) -> str:
            # Use a regular expression to split the string by whitespace and punctuation
            aprox_tokens = len(re.findall(r'\w+|\S', argument))
            if aprox_tokens > 6000:
                excess = ((aprox_tokens - 6000)/aprox_tokens)
                words = argument.split(' ') 
                argument = ' '.join(words[:int(len(words)*(1-excess))])
            return argument
            
        try:
            argument = truncate_argument(argument)
            payload = {
                "model": "llama3",
                "prompt": argument, 
                "system": "",
                "stream": False
            }
            payload_json = json.dumps(payload)
            headers = {"Content-Type": "application/json"}
            response = requests.post(OLLAMA_ENDPOINT, data=payload_json, headers=headers)
            if response.status_code == 200:
                return json.loads(response.text)["response"]
            else:
                return f"Ollama LLM API Error: {response.status_code}, {response.text}"
        except Exception as e:
            return f"Unexpected error calling the Ollama LLM API. Error: {str(e)}."
            
    def bind(self, stop=None):
        return SkipOrOp()
    
local_llm_tool = LocalLLMTool()

class LocalFilesSearchTool(BaseTool):
    name: str = "LocalFilesSearchTool"
    description: str = "Indexes all files in the local `data` directory."

    def _run(self) -> str:
        self.index = {}
        for root, _, files in os.walk('data'):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.index[file_path] = f.read()
        return "Directory indexed."

    def search(self, query: str):
        return [path for path, content in self.index.items() if query in content]
directory_search_tool = LocalFilesSearchTool()

class FileReadTool(BaseTool):
    name: str = "FileReadTool"
    description: str = "Reads the content of a specified file."

    def _run(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
file_read_tool = FileReadTool()

# Configure custom logging
logging.basicConfig(filename='custom_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_step_callback(step_context):
    log_message = (
        f"Task: {step_context.task_description}, "
        f"Agent: {step_context.agent_role}, "
        # f"Output: {step_context.output}"
    )
    logger_crewai.info("[Log] " + log_message)


search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

acceptance_criteria = (
    "1. The report 2 sections: '1. Summary' and '2. Details'.\n "
    "2. The Summary section have a table with the top universities selected ranked by total "
    "costs in descending order.\n "
    "3. The Details section have detailed information about each of the top universities, including:\n "
    "  - breaking down of main compomentes of cost of living.\n "
    "  - univerisity fees.\n "
    "  - application details.\n "
    "  - acceptance rate.\n ")

# Agent 1: University Finder
university_finder = Agent(
    role="University Finder",
    goal="Identify and list the top universities in Europe offering theoretical physics programs with a focus on quantum physics.",
    tools=[local_llm_tool, scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As a University Finder, your expertise in searching and evaluating academic institutions "
        "ensures that only the top universities for theoretical physics are listed."
    ),
    allow_delegation=False,
)

# Agent 2: Application Researcher
application_analyst = Agent(
    role="Application Researcher",
    goal="Analyse and compare European universisies from many different aspects (fees, living costs, relevance, etc) "
         "and write a report comparing them.",
    tools=[local_llm_tool, scrape_tool, search_tool],
    verbose=True,
    backstory=
        "As an Application Researcher, you specialize in coordinate the team to analyze universities "
        "and produce a comprehensive report comparing all of them."
    ,
    allow_delegation=True,
)


# Agent 3: Application Analyst
application_analyst = Agent(
    role="Application Analyst",
    goal="Analyze and summarize all the information collected from any specific university.",
    tools=[local_llm_tool, scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As an Application Analyst, you specialize in dissecting the application processes "
        "of universities, making sure every requirement and deadline is clearly understood."
    ),
    allow_delegation=False,
)

# Agent 4: Cost Analyst
cost_analyst = Agent(
    role="University Cost Analyst",
    goal="Analyze university fees and any available incentives for Brazilian applicants in USD.",
    tools=[local_llm_tool, scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As a Cost Analyst, your ability to break down university fees and identify financial aids "
        "or incentives helps ensure that education remains affordable."
    ),
    allow_delegation=False,
)

# Agent 5: Living Cost Analyst
living_cost_analyst = Agent(
    role="Living Cost Analyst",
    goal="Analyze living costs in the cities where the universities are located.",
    tools=[local_llm_tool, scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As a Living Cost Analyst, you provide detailed insights into the cost of living in various cities, "
        "covering all necessary aspects from housing to healthcare."
    ),
    allow_delegation=False,
)

# Agent 6: Quantum Physics Relevance Analyst
quantum_relevance_analyst = Agent(
    role="Quantum Physics Relevance Analyst",
    goal="Analyze the relevance of the universities in the field of quantum physics by examining "
         "researchers and published articles.",
    tools=[local_llm_tool, scrape_tool, search_tool],
    verbose=True,
    backstory="As a Quantum Physics Relevance Analyst, your deep dive into the academic contributions "
              "and researchers at universities ensures that the institutions chosen are leaders in the field.",
    allow_delegation=False,    
)

# Agent 7: Research Quality Assurance Specialist
quality_assurance_analyst = Agent(
    role="Research Quality Assurance Specialist",
    goal="Analyze the final report and check if it meets the following acceptance criteria:\n {acceptance_criteria}",
    tools=[local_llm_tool, scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As a Research Quality Assurance Specialist, you ensure that the final report meets all specified criteria, "
        "providing a comprehensive and reliable overview of the universities."
    ),
    allow_delegation=False,
)





# Create 1st Crew
university_research_crew = Crew(
    agents=[
        university_finder
    ],
    tasks=[
        identify_universities_task,
    ],
    verbose=False,
    memory=True,
    process=Process.sequential,
)

inputs_details = {
    'user': 'Bruno',
    'top_k_universities': 100,
    'region': 'Europe',
}

# Run the first crew to list the universities if the list isn't already saved
result = None
if os.path.exists(f"data/{VERSION}_universities.json"):
    with open(f"data/{VERSION}_universities.json", 'r') as f:
        result = json.load(f)
else:
    # run the first crew to list the universities
    result = university_research_crew.kickoff(inputs=inputs_details)
    # save the result to a file
    with open(f"data/{VERSION}_universities.json", 'w') as f:
        json.dump(result.dict(), f, indent=4)


T = TypeVar('T', bound=BaseModel)
def str_to_model(data: str, model: Type[T]) -> T:
    data_dict = json.loads(data)
    return model.parse_obj(data_dict)

universities = str_to_model(result, UniversityList).universities
print(len(universities), "universities")
print(universities[0])

countries = set([u.country for u in universities])
print(len(countries), 'countries:')
print(countries)

# tasks result schema:
class UniversityFee(BaseModel):
    anual_fee_usd: float
    fee_details: str | None

class UniversityLivingCosts(BaseModel):
    anual_living_cost_usd: float
    living_costs_details: str | None

class UniversityRelevance(BaseModel):
    level: Literal['low', 'medium', 'high']
    main_researchers: str | None
    main_research_field_in_quantum_physics: str | None

class UniversityApplicatnsTestimonies(BaseModel):
    success_testimonies: str | None
    failure_testimonies: str | None

class UniversityDetails(BaseModel):
    details: Union[University, UniversityFee, UniversityLivingCosts, UniversityRelevance, UniversityApplicatnsTestimonies]
    
def run_auxiliary_crews(universities):
    results = []
    for i, university in enumerate(universities):        
        sub_tasks = []
        filename_prefix = f"{university['city']}-{university['country']}"
        filename_prefix = re.sub(r'[^a-zA-Z0-9-_]', '', filename_prefix)

        # Sub-Task 2.1: Analyze University Fees
        analyze_fees_task = Task(
            description=(
                f"Analyze the university fees involved for a Major in Phisics at {university.city}/{university.country} university "
                "and check for any incentives available for Brazilian applicants."
            ),
            expected_output=f"The anual {university.city}/{university.country} fee in USD and the detailed breakdown of university fees in "
                            "the `fee_details` property, with the main cost components and info about any available "
                            "financial incentives with particular attention to specific details that may matter "
                            "for Brazilian applicants, in a JSON format {'anual_fee_usd': 15000.00, 'fee_details': 'details ...'}.",
            agent=cost_analyst,
            tools=[local_llm_tool, scrape_tool, search_tool],
            output_json=UniversityFee,
            context=[],
            async_execution=True,
            output_file=f"data/{VERSION}_{filename_prefix}_fees.txt",
            step_callback=log_step_callback,
            max_iterations=1000,
            max_time_limit=60*60*24,  # 24 hours (for test)
        )
        sub_tasks.append(analyze_fees_task)

        # Sub-Task 2.2: Analyze Living Costs
        analyze_living_cost_task = Task(
            description=(
                f"Analyze living costs in {university.city}/{university.country} near {university.name}, "
                "including housing, food, leisure, and healthcare. Pay attention to and flag any restrictions "
                "for applicants from Brazil."
            ),
            expected_output=f"The anual living costs in USD at {university.city}/{university.country} in a JSON format "
                            "like in `{'anual_living_cost_usd': 15000.00, 'living_costs_details': 'details ...'}`. "
                            "The detailed breakdown of living costs fees should cover: housing, food, health, pleasure, etc.",
            agent=living_cost_analyst,
            tools=[local_llm_tool, scrape_tool, search_tool],
            output_json=UniversityLivingCosts,
            context=[],
            async_execution=True,
            output_file=f"data/{VERSION}_{filename_prefix}_living_costs.txt",
            step_callback=log_step_callback,
            max_iterations=1000,
            max_time_limit=60*60*24,  # 24 hours (for test)
        )
        sub_tasks.append(analyze_living_cost_task)

        # Task 2.3: Analyze Quantum Physics Relevance
        analyze_relevance_task = Task(
            description=(
                f"Analyze the relevance of the university {university.city}/{university.country} in quantum physics by examining the main researchers "
                "in the field and the main topics in the field with relevant articles published in the last 5 years."
            ),
            expected_output=f"An analysis of the {university.city}/{university.country} university's relevance in quantum physics based on researchers and publications. "
                            "Fill the list of main top 3 researchers and research topics in the `main_researchers` and "
                            "`main_research_field_in_quantum_physics` fields respectively. "
                            "Fill the `level` field according your assessment.",
            agent=quantum_relevance_analyst,
            tools=[local_llm_tool, scrape_tool, search_tool],
            output_json=UniversityRelevance,
            context=[],
            async_execution=True,
            step_callback=log_step_callback,
            output_file=f"data/{VERSION}_{filename_prefix}_living_costs.txt",
            max_iterations=1000,
            max_time_limit=60*60*24,  # 24 hours (for test)
        )
        sub_tasks.append(analyze_relevance_task)

        # Task 2.4: Add testimonies
        add_testimonies_task = Task(
            description=(
                f"Retrive testimonies from other past applicants to {university.city}/{university.country} that succeed and also failed. "
                "Search for testimonies of both success and failures of similar applicants worldwide, "
                "especially those from South America and Brazil. "
            ),
            expected_output=f"Each testimony of {university.city}/{university.country} applicants should be summarized with its most inspiring and informative highlights. "
                            "Always add a link with reference to the source of the testimony. "
                            f"And make sure to comply with the following acceptance criteria:\n {acceptance_criteria}",
            agent=application_analyst,
            tools=[local_llm_tool, scrape_tool, search_tool],
            output_json=UniversityApplicatnsTestimonies,
            context=[],
            async_execution=True,
            output_file=f"data/{VERSION}_{filename_prefix}_testimonies.txt",
            step_callback=log_step_callback,
            max_iterations=1000,
            max_time_limit=60*60*24,  # 24 hours (for test)
        )
        sub_tasks.append(add_testimonies_task)

        # Task 2: Analyze Application Requirements
        analyze_application_task = Task(
            description=(
                f"Review and aggregate all information gathered about {university.city}/{university.country} university "
                "for a application in one JSON object, combining fees, living costs, relevance in quantum phisics and testimonies."
            ),
            expected_output=f"Combine the information of the university {university.city}/{university.country} and "
                            "the details about university fees, living costs, and relevance in the field in "
                            "one object that aggregates all properties in one JSON with the same property names.",
            agent=application_analyst,
            tools=[local_llm_tool, directory_search_tool, file_read_tool],
            output_json=UniversityDetails,
            context=sub_tasks, # all previous sub-tasks are required
            async_execution=False,
            step_callback=log_step_callback,
            output_file=f"data/{VERSION}_{filename_prefix}_all_details.json",
            max_iterations=1000,
            max_time_limit=60*60*24,  # 24 hours (for test)
        )

        # Task 3: Build a final report in markdown format for this university
        build_partial_report_task = Task(
            description=(
                f"Aggregate the collected information by the other team members about {university.city}/{university.country} , "
                "and write a markdown comprehensive report with all relevant information collected by the other agents."
            ),
            expected_output=f"A final mardown report about {university.city}/{university.country} covering, if possible, "
                            f"all items mentioned in the acceptance criteria:\n {acceptance_criteria}",
            agent=application_analyst,
            tools=[local_llm_tool, directory_search_tool, file_read_tool],
            context=[analyze_application_task],
            output_file=f"data/{VERSION}_{filename_prefix}_report.md",
            async_execution=False,
            step_callback=log_step_callback,
            max_iterations=20,
            max_time_limit=60*60*1,  # 1 hour (for test)
        )

        # Create Crew to process new tasks/sub-tasks
        university_research_crew_final = Crew(
            agents=[
                university_finder, 
                application_analyst, 
                cost_analyst, 
                living_cost_analyst, 
                quantum_relevance_analyst],
            tasks=sub_tasks+[analyze_application_task, build_partial_report_task],
            verbose=False,
            memory=True,
            process=Process.sequential,
        )

        report = university_research_crew_final.kickoff(inputs=inputs_details)
        results.append({'university': university, 'report': report})

    return results

N_UNIVERSITIES = 2
list_of_universities = ' '.join([f"{u.name} ({u.country})," for u in universities[:N_UNIVERSITIES]])[:-1]
print('Researching:', list_of_universities)

universities_reports = run_auxiliary_crews(universities[:N_UNIVERSITIES])


# Task 4: Build a final "Top K Universities for Bruno" report in markdown format
build_final_report_task = Task(
    description=(
        "Aggregate the collected information by the other team members about ALL universities analyzed, "
        "and improve a markdown report ranking them, adding the new university to the existing report "
        f"with all information collected by the other agents. The universities are: {list_of_universities}."
    ),
    expected_output="The final and comprehensive report 'Europe University Analyses', with the most recommended for {user} first, divided into two sections. "
                    "First, a summary, with a table showing: City (Country), University, University Fee, Living Costs, Application Summary, Testimonies. "
                    f"And make sure to comply with the following acceptance criteria:\n {acceptance_criteria}",
    agent=application_analyst,
    tools=[local_llm_tool],
    context=universities_reports,
    output_file="top_universities_azure_single_crew.md",
    async_execution=False,
    step_callback=log_step_callback,
    max_iterations=20,
    max_time_limit=60*60*1,  # 1 hour (for test)
)

# Create Crew to process new tasks/sub-tasks
university_research_crew_final = Crew(
    agents=[
        university_finder, 
        application_analyst, 
        cost_analyst, 
        living_cost_analyst, 
        quantum_relevance_analyst],
    tasks=[build_final_report_task],
    verbose=False,
    memory=True,
    process=Process.sequential,
)

final_result = university_research_crew_final.kickoff(
    inputs={'user': 'Bruno', 'universities': list_of_universities})

print()
print('DONE!')
print()
print(final_result)

