from langchain import OpenAI
from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer
from langchain import SerpAPIWrapper

llm = OpenAI(
    openai_api_key="OPENAI_API_KEY",
    temperature=0,
    model_name="text-davinci-003"
)

llm_math = LLMMathChain(llm=llm)

# initialize the math tool
math_tool = Tool(
    name='Calculator',
    func=llm_math.run,
    description='Useful for when you need to answer questions about math.'
)
# when giving tools to LLM, we must pass as list of tools
tools = [math_tool]

tools = load_tools(
    ['llm-math'],
    llm=llm
)

zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3
)

# zero_shot_agent("what is (4.5*2.1)^2.2?")

# zero_shot_agent("if Mary has four apples and Giorgio brings two and a half apple "
#                 "boxes (apple box contains eight apples), how many apples do we "
#                 "have?")

# zero_shot_agent("what is the capital of Norway?") # how to fix this error?

prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# initialize the LLM tool
llm_tool = Tool(
    name='Language Model',
    func=llm_chain.run,
    description='use this tool for general purpose queries and logic'
)

# tools.append(sql_tool)

zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
)

memory = ConversationBufferMemory(memory_key="chat_history")
conversational_agent = initialize_agent(
    agent='conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    memory=memory,
)
result = conversational_agent(
    "Please provide me the stock prices for ABC on January the 1st"
)

docstore=DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description='search wikipedia'
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description='lookup a term in wikipedia'
    )
]

docstore_agent = initialize_agent(
    tools,
    llm,
    agent="react-docstore",
    verbose=True,
    max_iterations=3
)

# initialize the search chain
search = SerpAPIWrapper(serpapi_api_key='serp_api_key')

# create a search tool
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description='google search'
    )
]

# initialize the search enabled agent
self_ask_with_search = initialize_agent(
    tools,
    llm,
    agent="self-ask-with-search",
    verbose=True
)

