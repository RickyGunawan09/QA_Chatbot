import streamlit as st
import function_call as fc

st.title("QA Chatbot Spotify Reviews")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = fc.function_call(prompt)
        response = st.write(result)
        st.session_state.messages.append({"role": "assistant", "content": response})