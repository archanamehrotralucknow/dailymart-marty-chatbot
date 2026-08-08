from io import BytesIO

import streamlit as st
from gtts import gTTS
from streamlit_mic_recorder import speech_to_text

import chatbot

st.set_page_config(page_title="Marty Voice", page_icon="🎙️")


def get_bot() -> chatbot.ChatBot:
    if "bot" not in st.session_state:
        st.session_state.bot = chatbot.ChatBot()
    return st.session_state.bot


def speak(text: str) -> bytes:
    buffer = BytesIO()
    gTTS(text=text, lang="en").write_to_fp(buffer)
    return buffer.getvalue()


def render_turn(turn: dict, autoplay: bool) -> None:
    with st.chat_message(turn["role"]):
        st.write(turn["text"])
        for card in turn.get("cards", []):
            st.markdown(f"**{card['title']}** — {card['merchant']} · {card['price']} · {card['rating']}  \n{card['affiliate_url']}")
        if turn.get("disclosure"):
            st.caption(turn["disclosure"])
        if turn.get("audio"):
            st.audio(turn["audio"], format="audio/mp3", autoplay=autoplay)


st.title("🎙️ Marty Voice")
st.caption("Dailymart's shopping assistant, powered by a local Ollama model.")

if "history" not in st.session_state:
    st.session_state.history = []
if "last_voice" not in st.session_state:
    st.session_state.last_voice = None

with st.sidebar:
    st.header("Settings")
    chatbot.OLLAMA_HOST = st.text_input("Ollama host", chatbot.OLLAMA_HOST).rstrip("/")
    chatbot.OLLAMA_MODEL = st.text_input("Ollama model", chatbot.OLLAMA_MODEL)
    speak_replies = st.toggle("Speak replies", value=True)
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.bot = chatbot.ChatBot()
        st.rerun()

st.write("Tap the mic and speak, or type below.")
spoken = speech_to_text(language="en", start_prompt="🎙️ Speak", stop_prompt="⏹️ Stop", key="mic")
typed = st.chat_input("Type a message")

for index, turn in enumerate(st.session_state.history):
    render_turn(turn, autoplay=False)

message = typed or (spoken if spoken and spoken != st.session_state.last_voice else None)
if spoken:
    st.session_state.last_voice = spoken

if message:
    response = get_bot().handle_message(message)
    st.session_state.history.append({"role": "user", "text": message})
    audio = speak(response["reply_text"]) if speak_replies else None
    st.session_state.history.append({
        "role": "assistant",
        "text": response["reply_text"],
        "cards": response["product_cards"],
        "disclosure": response["affiliate_disclosure"],
        "audio": audio,
    })
    render_turn(st.session_state.history[-2], autoplay=False)
    render_turn(st.session_state.history[-1], autoplay=True)
