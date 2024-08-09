import streamlit as st
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]


def rephraseAnswer(question, answer):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"""Rephrase the Answer into a meaningful sentence: "{answer}" 
            using the Question: "{question}" 
            Provide only the rephrased answer and nothing else."""}
        ],)

    return  response.choices[0].message["content"].strip()