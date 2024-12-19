import os
from glob import glob
from typing import List
from transformers.agents.default_tools import Tool
from transformers.agents import Toolbox
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from da_agent.agent.generatedtool import parse_generated_tools

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")
EMBED_MODEL_TYPE = os.getenv("EMBED_MODEL_TYPE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class ToolRetriever:
    def __init__(self, generated_tool_dir: str):
        self.generated_tool_dir = generated_tool_dir
        self.vectordb_path = os.path.join(self.generated_tool_dir, "vectordb")

        if not os.path.exists(self.vectordb_path):
            os.makedirs(self.vectordb_path)

        # 初始化向量数据库和嵌入函数
        # 根据您的实际情况，设置 embedding_function
     
        if EMBED_MODEL_TYPE == "OpenAI":
            embedding_function = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                model="text-embedding-3-large"
            )
            embed_model_name = EMBED_MODEL_NAME

        self.vectordb = Chroma(
            collection_name="tool_vectordb",
            embedding_function=embedding_function,
            persist_directory=self.vectordb_path,
        )

        self.generated_tools = {}
        for path in glob(os.path.join(generated_tool_dir, "*.py")):
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
            tools = parse_generated_tools(code)
            for tool in tools:
                self.generated_tools[tool.name] = tool

    def retrieve(self, query: str, k: int = 10) -> List[Tool]:
        k = min(len(self.vectordb), k)
        if k == 0:
            return []
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)
        tools = []
        for doc, _ in docs_and_scores:
            name = doc.metadata["name"]
            tools.append(self.generated_tools[name])
        return tools

    def add_new_tool(self, tool: Tool):
        program_name = tool.name
        program_description = tool.description

        res = self.vectordb._collection.get(ids=[program_name])
        # 检查工具是否已存在
        if res["ids"]:
            raise ValueError(f"Tool {program_name} already exists!")
            # 如果需要覆盖，可以删除并重新添加
            # self.vectordb._collection.delete(ids=[program_name])

        # 添加新工具到向量数据库和工具字典
        self.vectordb.add_texts(
            texts=[program_description],
            ids=[program_name],
            metadatas=[{"name": program_name}],
        )
        self.generated_tools[tool.name] = tool

        self.vectordb.persist()

    def add_new_tool_from_path(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        tools = parse_generated_tools(code)
        for tool in tools:
            self.add_new_tool(tool)


class ToolRetrievalTool(Tool):
    name = "get_relevant_tools"
    description = 'This tool retrieves relevant tools generated in previous runs. Provide a query describing what you want to do. If there are no tools in the toolbox, "No tool found" will be returned.'
    inputs = "query: str"
    output_type = "str"

    def __init__(self, generated_tool_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_tool_dir = generated_tool_dir
        self.tool_retriever = ToolRetriever(generated_tool_dir)
        self.tool_description_template = "<<tool_name>>: <<tool_description>>"

    def forward(self, query: str) -> str:
        relevant_tools: List[Tool] = self.tool_retriever.retrieve(query)
        if relevant_tools:
            relevant_toolbox = Toolbox(relevant_tools)
            return relevant_toolbox.show_tool_descriptions(
                self.tool_description_template
            )
        else:
            return "No tool found"

    def add_new_tool_from_path(self, path: str):
        return self.tool_retriever.add_new_tool_from_path(path)