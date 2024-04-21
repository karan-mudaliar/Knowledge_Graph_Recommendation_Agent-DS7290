import streamlit as st
from dotenv import load_dotenv
import os
from neo4j import GraphDatabase
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from openai import OpenAI
import threading
from datetime import time

# Load environment variables
load_dotenv()

# Neo4j connection setup
NEO4J_URL = os.getenv('NEO4J_URL')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
neo4j_driver = GraphDatabase.driver(
    uri=NEO4J_URL,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    database=NEO4J_DATABASE,
    connection_timeout=50*60
)

# Initialize OpenAI client
client = OpenAI()

# Import context generator
from query_generator import get_context


SYSTEM_PROMPT = """1)You are a helpful movie expert agent that is polite, who talks and helps users find something they like, recommend them more movies, learn more about movies, or find its rating, and only stick to english, NO OTHER LANGUAGES. 
                   2)You will be given user prompt along with some context to help you generate responses. Write recommendations based on the context. 
                   3)You will not answer anything outside movie, tv,actor or any related domain. If something outside this domain is asked you will say, I'm sorry but I cant help you with this. 
                   4)Always keeps the responses short and within 5-6 sentences, be nice and helpful
                   5)You will be given retrived data from a knowledge graph as context to help you.
                   6)ONLY THE USER_PART IS GIVEN BY THE USER, THE CONTEXT IS RETRIEVED FOR YOU SO THAT YOU CAN HELP THE USER.
                   7)Always act like the additional context is something you know and you are recommending it. be friendly and talk as an assistant, example - It's great to see you're interested in .....
                   8)If additional context has good suggestions, always use it in your recommendations.
                   9) Don't recommend or elaborate on anything not provided in the context
                   OUTPUT:
                   a) If asked about recommendations:
                    <text about what the user has mentioned, in a friendly conversational way, and a short summary>
                    **Recommendations:** \n
                        1. <movie_1>: A drama .........
                        2. <movie_2>: About .........
                   b) For anything else, just have a friendly conversation, guiding the user
                   """
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = Ollama(base_url='http://localhost:11434', model="wizardlm2", callbacks=callback_manager, system=SYSTEM_PROMPT)

conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=3),
    verbose=False
)

def get_response(user_input):
    context = get_context(neo4j_driver, client, user_input)
    prompt = f"USER_PART : {user_input} \n CONTEXT: {context}"
    response = conversation_with_summary.predict(input=prompt)
    return response

# # Streamlit App Interface
# st.title('Movie Assistant')
# user_input = st.text_input("Enter your query about movies, or type 'exit' to quit:")
# if user_input:
#     if user_input.lower() == 'exit':
#         st.write("Exiting the conversation...")
#     else:
#         response = get_response(user_input)
#         st.write("Response from LLM:", response)

st.title('Movie Assistant')
user_input = st.text_input("Enter your query about movies, or type 'exit' to quit:")
if user_input:
    if user_input.lower() == 'exit':
        st.write("Exiting the conversation...")
    else:
        response = get_response(user_input)
        output_container = st.empty()
        # Simulating streaming by splitting response and sending out words
        for word in response.split():
            output_container.write(f'{word} ')
            time.sleep(0.5)  # Delay between words, can be adjusted as needed
