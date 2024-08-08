import streamlit as st
import pandas as pd
from langchain import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import os
import time
from utils.language_utils import *

# Load CSS file
def load_css(file_name="static/style.css"):
    with open(file_name, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "df" not in st.session_state:
    st.session_state.df = None

def stream_data(input):
    for word in input.split(" "):
        yield word + " "
        time.sleep(0.08)

# Login Page
def login():
    load_css()
    logo_path = os.path.join("utils", "logo2.png")
    st.image(logo_path, width=200)
    st.title("Login to Multi-lingual Q&A Chatbot")
    st.markdown("###### Panchayat & Rural Development Dept, Bhopal, M.P")
    with st.form(key='login_form'):
        username = st.text_input("Username", placeholder="Enter your username here...")
        password = st.text_input("Password", type="password", placeholder="Enter your password here...")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if username == st.secrets["LOGIN_USERNAME"] and password == st.secrets["LOGIN_PASSWORD"]:
                st.session_state.authenticated = True
                st.success("Logged In as {}".format(username))
                st.rerun()
            else:
                st.error("Incorrect Username/Password")

# Main App
def main_app():
    load_css()
    st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">', unsafe_allow_html=True)

    logo_path = os.path.join("utils", "logo2.png")
    st.logo(logo_path, icon_image=logo_path)

    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    llm = OpenAI(temperature=0)

    st.title("Multi-lingual Q&A Chatbot")
    st.markdown("###### Panchayat & Rural Development Dept, Bhopal, M.P")

    if st.session_state.df is None:
        st.session_state.df = pd.read_csv("./MGNREGA_AGG.csv")
    
    if st.session_state.df is not None:
        agent = create_pandas_dataframe_agent(
            llm, st.session_state.df, verbose=True, allow_dangerous_code=True
        )
        
        if not st.session_state.chat_history:
            st.markdown(
                """
                <div style="display: flex; align-items: center; justify-content: center; height: 200px;">
                    <i class="fa-solid fa-message" style="font-size: 48px; color: rgb(224, 224, 224); margin-right: 10px;"></i>
                    <h2 style="font-size: 24px; font-weight: bold; color: grey; margin: 0;">Start your conversation</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_question = st.chat_input("Your question:")
        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            try:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = agent.run(user_question)
                    st.write_stream(stream_data(response))
                    
                    translated_text = translateText(response, src_lang='en', target_lang='hi')
                    st.write_stream(stream_data(translated_text))
                    
                    audio_b64 = textToSpeech(language='hi', text = translated_text)
                    audio_player(audio_b64)
                    
                    response_for_history = f'''{response}\n\n{translated_text}'''
                    st.session_state.chat_history.append({"role": "assistant", "content": response_for_history})
            except Exception as e:
                st.write("Unable to generate response, please try again.")
                st.error(f"Error: {str(e)}")

        with st.sidebar:
            st.markdown("### Data from your uploaded file:")
            st.dataframe(st.session_state.df)
    else:
        st.write("Please upload an Excel file to begin.")

# Main execution
if st.session_state.authenticated:
    main_app()
else:
    login()
