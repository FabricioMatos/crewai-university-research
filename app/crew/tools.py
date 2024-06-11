import os

from crewai_tools import (
    BaseTool,
    ScrapeWebsiteTool,
    SerperDevTool,
)

# TODO: this should be  a simpler tool, just getting all the reports together


class ReadAllPartialReportsTool(BaseTool):
    name: str = "ReadAllPartialReportsTool"
    description: str = (
        "Return all the partial reports in the `./data/partial` folder."
    )
    index: dict = {}

    def load_files(self):
        for root, _, files in os.walk("data"):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        self.index[file_path] = f.read()

    def _run(self) -> str:
        if not self.index:
            self.load_files()

        result: str = ""
        for file_path, content in self.index.items():
            result += f"<{file_path}>\n\n{content}\n\n</{file_path}>\n\n"

        return f"Here are all partial reports:\n\n{result}\n\n"


# tools for search and scrapping
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# access to local files with partial results
read_partial_reports_tool = ReadAllPartialReportsTool()
