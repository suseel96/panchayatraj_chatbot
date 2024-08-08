import requests
import json
import base64
import streamlit as st
from translators import translate_text

def autoplay_audio(b64):
# def autoplay_audio(file_path: str):
    # with open(file_path, "rb") as f:
    #     data = f.read()
    # b64 = base64.b64encode(data).decode("utf-8")
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
    # wav_file = open("temp.wav", "wb")
    # decode_string = base64.b64decode(base64_data)
    # wav_file.write(decode_string)