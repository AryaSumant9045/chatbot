import streamlit as st
# from connected_llm import  queryMe
from rajeev import main

def ask(query):
    return main(query)

st.title("Rajiv Dixit Chatbot")

query = st.chat_input("Write your query")


if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if query:
    with st.chat_message("user"):
        st.markdown(query)  
    st.session_state.messages.append({'role':'user', 'content':query})

    response = ask(query)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({'role':'assistant','content':response})
