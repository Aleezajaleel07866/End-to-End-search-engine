# importing libraries
import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun 
from langchain.agents import initialize_agent, AgentType 
from langchain.callbacks import StreamlitCallbackHandler
import os 
from dotenv import load_dotenv

## Initialize Arxiv and wikipedia tools
## creating an arxiv wrapper to define how many results to fetch and how many content from ech
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# creating aa DuckDuckGo search tool for general purpose search
search = DuckDuckGoSearchRun(name ="Search")


# streamlit UI Setup
st.title("🔎 Search Enginne (Gen AI app) using Tools and Agents")

# creating sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Please enter Groq Api key:", type="password")

# session state for chat message
# if message doesnt exit in session state, initialize with greeting message
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {
            "role":"assistant",
            "content" : "Hi, I am a chatbot who can serch the web. How can I help you?"
        }
    ]

#loop through all past messaged and display time in the chat window
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# chat input and preprocessing


#when user input a new message in chat
if prompt := st.chat_input(placeholder="What is machine learning? "):
    #append user's message to session staate message list
    st.session_state.messages.append({"role":"user","content": prompt})
    # show the user's message in the chat interface
    st.chat_message("user").write(prompt)

    # initiaalized chatgroq LLM using provided API
    llm = ChatGroq(groq_api_key = api_key,model_name = "llama-3.1-8b-instant")

    tools = [search,arxiv,wiki]

    # we will creaate an agent that uses .ZERO_SHOT_REACT_DECRIPTION
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors = True
    )

    # get and display the response

    with st.chat_message("assitant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts =False)

        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])

        st.session_state.messages.append({
            'role':'assistant',
            'content': response
        })

        #display the assitant's response in the chat ui
        st.write(response)



