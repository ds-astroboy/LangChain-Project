import os
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import OpenAI, AzureOpenAI

os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'HUGGINGFACEHUB_API_KEY'

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
        template=template,
    input_variables=['question']
)


# initialize Hub LLM
# hub_llm = HuggingFaceHub(
#         repo_id='google/flan-t5-xl',
#     model_kwargs={'temperature':1e-10}
# )
#
# # create prompt template > LLM chain
# llm_chain = LLMChain(
#     prompt=prompt,
#     llm=hub_llm
# )

########################### single question template prompt

# # user question
question = "Which NFL team won the Super Bowl in the 2010 season?"
#
# # ask the user question about NFL 2010
# print(llm_chain.run(question))

# qs = [
#     {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
#     {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
#     {'question': "Who was the 1st person on the moon?"},
#     {'question': "How many eyes does a blade of grass have?"}
# ]
# res = llm_chain.run(qs)


########################### Multiple question template prompt

# # initialize HF LLM
# flan_t5 = HuggingFaceHub(
#     repo_id="google/flan-t5-xl",
#     model_kwargs={"temperature":1e-10}
# )
#
# multi_template = """Answer the following questions one at a time.
#
# Questions:
# {questions}
#
# Answers:
# """
# long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])
#
# llm_chain = LLMChain(
#     prompt=long_prompt,
#     llm=flan_t5
# )
#
# qs_str = (
#     "Which NFL team won the Super Bowl in the 2010 season?\n" +
#     "If I am 6 ft 4 inches, how tall am I in centimeters?\n" +
#     "Who was the 12th person on the moon?" +
#     "How many eyes does a blade of grass have?"
# )
#
# print(llm_chain.run(qs_str))

############################# OpenAI API

os.environ['OPENAI_API_KEY'] = 'OpenAI_API_KEY'

davinci = OpenAI(model_name='text-davinci-003')

llm_chain = LLMChain(
    prompt=prompt,
    llm=davinci
)

print(llm_chain.run(question))

qs = [
    {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
    {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
    {'question': "Who was the 12th person on the moon?"},
    {'question': "How many eyes does a blade of grass have?"}
]
llm_chain.generate(qs)

qs = [
    "Which NFL team won the Super Bowl in the 2010 season?",
    "If I am 6 ft 4 inches, how tall am I in centimeters?",
    "Who was the 12th person on the moon?",
    "How many eyes does a blade of grass have?"
]
print(llm_chain.run(qs))

multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""
long_prompt = PromptTemplate(
    template=multi_template,
    input_variables=["questions"]
)

llm_chain = LLMChain(
    prompt=long_prompt,
    llm=davinci
)

qs_str = (
    "Which NFL team won the Super Bowl in the 2010 season?\n" +
    "If I am 6 ft 4 inches, how tall am I in centimeters?\n" +
    "Who was the 12th person on the moon?" +
    "How many eyes does a blade of grass have?"
)

print(llm_chain.run(qs_str))