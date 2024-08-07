import streamlit as st
import pandas as pd
from langchain import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import os


# Load CSS file
def load_css(file_name="static/style.css"):
    with open(file_name, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css()


logo_path = os.path.join("utils", "logo2.png")

st.logo(logo_path, icon_image=logo_path)
# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for the dataframe
if "df" not in st.session_state:
    st.session_state.df = None

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

llm = OpenAI(temperature=0)

# Streamlit app
st.title("Multi-lingual Q&A Chatbot")
st.markdown("###### Panchayat & Rural Development Dept, Bhopal, M.P")

st.session_state.df = pd.read_csv("./MGNREGA_AGG.csv")

if st.session_state.df is not None:
    # Create a pandas dataframe agent
    agent = create_pandas_dataframe_agent(
        llm, st.session_state.df, verbose=True, allow_dangerous_code=True
    )

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_question = st.chat_input("Your question:")

    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Display user message
        with st.chat_message("user"):
            st.write(user_question)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.run(user_question)
            st.write(response)

        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display the dataframe
    with st.sidebar:
        st.markdown("### Data from your uploaded file:")
        st.dataframe(st.session_state.df)

else:
    st.write("Please upload an Excel file to begin.")
