from dotenv import load_dotenv
import streamlit as st
import os
from neo4j import GraphDatabase
from langchain_community.llms import Ollama 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
load_dotenv()

# Update this import based on the actual location and definition
from query_generator import get_context
from neo4j import GraphDatabase


load_dotenv()

# Setting up Neo4j connection
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

client = OpenAI()

# Setting up the LLM with LangChain
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
                    <text about what the user has mentioned, in a friendly conversational way, and a short summary> \n
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

def main_loop():
    while True:
        user_input = input("Enter your query or type 'exit' to quit: ")
        if user_input.lower() == 'exit':
            print("Exiting the conversation...")
            break
        
        context = get_context(neo4j_driver, client, user_input)  # Ensure this function call matches your setup
        prompt = f"USER_PART : {user_input} \n CONTEXT: {context}"
        response = conversation_with_summary.predict(input=prompt)
        print("Response from LLM:", response)

if __name__ == "__main__":
    main_loop()

# st.title("Movie Expert Chat Agent")
# user_input = st.text_input("Enter your movie query:")

# if st.button("Send"):
#     with st.spinner("Fetching response..."):
#         context = get_context(neo4j_driver, client, user_input)  # Ensure this function call matches your setup
#         prompt = f"USER_PART : {user_input} \n CONTEXT: {context}"

#         # Set up the LLM with LangChain
#         callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#         llm = Ollama(base_url='http://localhost:11434', model="wizardlm2", callbacks=callback_manager, system=SYSTEM_PROMPT)

#         # Create conversation chain
#         conversation_with_summary = ConversationChain(
#             llm=llm,
#             memory=ConversationBufferWindowMemory(k=3),
#             verbose=False
#         )

#         # Predict the response using LangChain
#         response = conversation_with_summary.predict(input=prompt)
#         st.write("Response from LLM:", response)

# # Display an exit button to stop the session
# if st.button("Exit"):
#     st.write("Exiting the conversation...")
#     st.stop()