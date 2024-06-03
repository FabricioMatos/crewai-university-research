"""
The MIT License (MIT)
Copyright © 2024 Fabricio Vargas Matos

Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files (the “Software”), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software 
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import json
import re
import requests

from crewai_tools import (
    BaseTool,
    ScrapeWebsiteTool,
    SerperDevTool,
)

OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
VERSION = "v2_llama3"


class SkipOrOp:
    def run(self, *args, **kwargs):
        return None

    def __or__(self, other):
        return other


class LocalLLMTool(BaseTool):
    name: str = "[Default tool] Ollama Local LLM chat tool"
    description: str = (
        "A tool to interact with a local LLM served by Ollama. "
        "Expets a string as input."
    )

    def _run(self, argument: str) -> str:
        def truncate_argument(argument: str) -> str:
            # Use a regex to split the string by whitespace/punctuation
            aprox_tokens = len(re.findall(r"\w+|\S", argument))
            if aprox_tokens > 6000:
                excess = (aprox_tokens - 6000) / aprox_tokens
                words = argument.split(" ")
                argument = " ".join(words[: int(len(words) * (1 - excess))])
            return argument

        try:
            argument = truncate_argument(argument)
            payload = {
                "model": "llama3",
                "prompt": argument,
                "system": "",
                "stream": False,
            }
            payload_json = json.dumps(payload)
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                OLLAMA_ENDPOINT, data=payload_json, headers=headers
            )
            if response.status_code == 200:
                return json.loads(response.text)["response"]
            else:
                return (
                    f"Ollama LLM API Error: "
                    f"{response.status_code}, {response.text}"
                )
        except Exception as e:
            return (
                f"Unexpected error calling the Ollama LLM API. "
                f"Error: {str(e)}."
            )

    def bind(self, stop=None):
        return SkipOrOp()


class LocalFilesSearchTool(BaseTool):
    name: str = "LocalFilesSearchTool"
    description: str = "Indexes all files in the local `data` directory."

    def _run(self) -> str:
        self.index = {}
        for root, _, files in os.walk("data"):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    self.index[file_path] = f.read()
        return "Directory indexed."

    def search(self, query: str):
        return [path for path, content in self.index.items() if query in content]


class FileReadTool(BaseTool):
    name: str = "FileReadTool"
    description: str = "Reads the content of a specified file."

    def _run(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


# tools for search and scrapping
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# local LLM (llama3-8B with ollama)
local_llm_tool = LocalLLMTool()

# access to local files with partial results
directory_search_tool = LocalFilesSearchTool()
file_read_tool = FileReadTool()
