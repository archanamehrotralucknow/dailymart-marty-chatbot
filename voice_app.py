from io import BytesIO

import streamlit as st
from gtts import gTTS
from streamlit_mic_recorder import speech_to_text

import chatbot

st.set_page_config(page_title="Marty", layout="centered")

STYLE = """
<style>
@keyframes pulse { 0% { transform: scale(1); opacity: .9; } 50% { transform: scale(1.4); opacity: .4; } 100% { transform: scale(1); opacity: .9; } }
@keyframes blink { 0%, 80%, 100% { opacity: .2; } 40% { opacity: 1; } }
.status-line { display: flex; align-items: center; gap: .55rem; color: #6b7280; font-size: .9rem; }
.status-line .dot { width: .7rem; height: .7rem; border-radius: 50%; background: #ef4444; animation: pulse 1.2s ease-in-out infinite; }
.thinking span { display: inline-block; width: .45rem; height: .45rem; margin: 0 .1rem; border-radius: 50%; background: #9ca3af; animation: blink 1.4s infinite both; }
.thinking span:nth-child(2) { animation-delay: .2s; }
.thinking span:nth-child(3) { animation-delay: .4s; }
</style>
"""

LISTENING = '<div class="status-line"><span class="dot"></span>Ready to listen</div>'
THINKING = '<div class="status-line"><span class="thinking"><span></span><span></span><span></span></span> Marty is thinking</div>'


def secret(name: str, fallback: str) -> str:
    """Read a Streamlit secret, falling back when it is missing OR blank."""
    try:
        value = str(st.secrets[name]).strip()
    except Exception:
        return fallback
    return value or fallback


# Streamlit secrets win over the .env value chatbot.py loads at import time.
chatbot.GEMINI_API_KEY = secret("GEMINI_API_KEY", chatbot.GEMINI_API_KEY)


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
            with st.container(border=True):
                st.markdown(f"**{card['title']}**")
                st.write(f"{card['merchant']}  ·  {card['price']}  ·  {card['rating']}")
                st.link_button("View deal", card["affiliate_url"])
        if turn.get("disclosure"):
            st.caption(turn["disclosure"])
        if turn.get("audio"):
            st.audio(turn["audio"], format="audio/mp3", autoplay=autoplay)


st.markdown(STYLE, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "last_voice" not in st.session_state:
    st.session_state.last_voice = None

with st.sidebar:
    st.subheader("Settings")
    chatbot.GEMINI_MODEL = st.text_input("Gemini model", secret("GEMINI_MODEL", chatbot.GEMINI_MODEL))
    chatbot.GEMINI_TIMEOUT = st.slider("Response timeout (s)", 15, 180, chatbot.GEMINI_TIMEOUT, 5)
    if chatbot.GEMINI_API_KEY:
        st.caption(f"API key loaded (…{chatbot.GEMINI_API_KEY[-4:]})")
    else:
        st.warning("No GEMINI_API_KEY found in .env or Streamlit secrets.")
    speak_replies = st.toggle("Speak replies", value=True)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.bot = chatbot.ChatBot()
        st.rerun()

    st.divider()
    if st.button("Check AI connection", use_container_width=True):
        st.session_state.conn = chatbot.check_connection()
    conn = st.session_state.get("conn")
    if conn:
        if conn["ok"] and chatbot.GEMINI_MODEL in conn["models"]:
            st.success(f"Connected · {len(conn['models'])} model(s) available")
        elif conn["ok"]:
            st.warning(
                f"Key works, but '{chatbot.GEMINI_MODEL}' isn't in this key's model list. "
                "Pick a model your key has access to."
            )
        else:
            st.error(f"Gemini unreachable: {conn['reason']}")

st.title("Marty")
st.caption("Dailymart shopping assistant, powered by the Gemini API.")

for turn in st.session_state.history:
    render_turn(turn, autoplay=False)

left, right = st.columns([1, 4])
with left:
    spoken = speech_to_text(language="en", start_prompt="Speak", stop_prompt="Stop", key="mic")
with right:
    st.markdown(LISTENING, unsafe_allow_html=True)

typed = st.chat_input("Ask about products, deals, or affiliate links")

message = typed or (spoken if spoken and spoken != st.session_state.last_voice else None)
if spoken:
    st.session_state.last_voice = spoken

if message:
    st.session_state.history.append({"role": "user", "text": message})
    render_turn(st.session_state.history[-1], autoplay=False)

    with st.chat_message("assistant"):
        status = st.empty()
        status.markdown(THINKING, unsafe_allow_html=True)
        response = get_bot().handle_message(message)
        audio = None
        if speak_replies:
            status.markdown('<div class="status-line">Generating voice</div>', unsafe_allow_html=True)
            audio = speak(response["reply_text"])
        status.empty()

    st.session_state.history.append({
        "role": "assistant",
        "text": response["reply_text"],
        "cards": response["product_cards"],
        "disclosure": response["affiliate_disclosure"],
        "audio": audio,
    })
    render_turn(st.session_state.history[-1], autoplay=True)
