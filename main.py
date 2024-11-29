from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_csv_agent
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize prompt template, llm, tools
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

tools = []  # Tools for chat bot

# Create tool for retrieving inventory data
## Load data
inventory_loader = CSVLoader(
    file_path="inventory_dataset.csv", 
    csv_args={
        "delimiter": ",",
        "fieldnames": ["Product Name", "Colour", "Cost Price", "Selling Price", "Quantities"],
    },
    encoding='utf-8',
)

inventory_data = inventory_loader.load()
credit_loader = CSVLoader("credit_limit.csv", encoding='utf-8')
credit = credit_loader.load()

## Embedding data
vector1 = FAISS.from_documents(inventory_data, OpenAIEmbeddings())
retriever1 = vector1.as_retriever()
vector2 = FAISS.from_documents(credit, OpenAIEmbeddings())
retriever2 = vector2.as_retriever()

## Create tool
inventory_retriever_tool = create_retriever_tool(
    retriever1,
    "inventory_search",
    "Search for information of cables in the inventory such as name, price, description, quantities."
    "Specify number of items to retrieve",
)
tools.append(inventory_retriever_tool)

credit_retriever_tool = create_retriever_tool(
    retriever2,
    "credit_search",
    "Search for information of credit limit of each customer",
)
tools.append(credit_retriever_tool)

def count_product(df):
    return len(df)

# Create agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template,
)
# agent = create_csv_agent(
#     ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
#     ["inventory_dataset.csv", "credit_limit.csv"],
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     allow_dangerous_code=True
# )
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run agent
agent_executor.invoke({"input": "How many products are there?"})
# agent.run("How many products are there?")