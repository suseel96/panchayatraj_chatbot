import requests
import json
import base64
import streamlit as st
from translators import translate_text
from langdetect import detect

def audio_player(audio_b64):
    audio_html = f"""
    <audio id="audio" controls>
        <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    <script>
    var audio = document.getElementById('audio');
    </script>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def autoplay_audio(b64):
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)


def translateText(text, src_lang, target_lang):
    translated_text = translate_text(translator = 'bing', query_text=text, from_language=src_lang, to_language=target_lang)
    return translated_text

def textToSpeech(language, text, voice = 'female'):

    url = "https://demo-api.models.ai4bharat.org/inference/tts"
    payload = json.dumps({
    "controlConfig": {
        "dataTracking": True
    },
    "input": [
        {
        "source": text}
    ],
    "config": {
        "gender": voice,
        "language": {
        "sourceLanguage": language
        }
    }
    })
    headers = {
    'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)


    base64_data = response.json()['audio'][0]['audioContent']
    return base64_data

def detectInputLang(input_text):
    return detect(input_text)