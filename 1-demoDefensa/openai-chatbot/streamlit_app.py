# Code refactored from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

import streamlit as st
from api_call import askQuestion

with st.sidebar:
    st.title('💬 Consultas tránsito')
#    if 'OPENAI_API_KEY' in st.secrets:
    #st.success('API key already provided!', icon='✅')
#        openai.api_key = st.secrets['OPENAI_API_KEY']
#    else:
 #       openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
   # st.warning('Please enter your credentials!', icon='⚠️')
   # st.success('Proceed to entering your prompt message!', icon='👉')

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
        message_placeholder = st.empty()
        full_response = ""
        full_response = askQuestion(prompt)
     #   for response in openai.ChatCompletion.create(
      #      model="gpt-3.5-turbo",
       #     messages=[{"role": m["role"], "content": m["content"]}
        #              for m in st.session_state.messages], stream=True):
         #   full_response += response.choices[0].delta.get("content", "")
          #  message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
