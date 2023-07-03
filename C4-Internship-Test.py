from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
import constants

db = SQLDatabase.from_uri("sqlite:///data/sqlite.db")
llm=OpenAI(openai_api_key=constants.OPENAI_API_KEY, temperature=0, model_name="gpt-3.5-turbo-0613")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# agent_executor.run("Records of applicants table refer to the job seekers for a certain internship. This internship is set to 3 months and for a full-time work from home. If this is not available, applicants should be ignored. This internship requires mainly Machine Learning, Natural Language Processing, Deep Learning. Based on this information and using information from applicants table determine 5 job seekers best matching to this job.")
# 21, 163, 237, 247, and 423

# agent_executor.run('Table applicants shows information for applicants applying for a internship job. List 5 applicants best matching to this job.')
# 21, 237, 247, 423, 466

agent_executor.run("Records of applicants table refer to the job seekers for a certain internship. This internship is set to 3 months and for a full-time work from home. If this is not available, applicants should be ignored. This internship requires mainly Machine Learning, Natural Language Processing, Deep Learning. Based on this information and using information from applicants table determine 5 job seekers best matching to this job.")


