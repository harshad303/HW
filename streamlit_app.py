import streamlit as st
st.set_page_config(page_title= "HW -IST 688")

# Show title and description.
st.title("HW Manager")

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

hw1_page = st.Page("hw1.py", title="HW1")
hw2_page = st.Page("hw2.py", title="HW2")
hw3_page = st.Page("hw3.py", title="HW3")
hw5_page = st.Page("hw5.py", title="HW5")
hw6_page = st.Page("hw6.py", title="HW6", default= True)
pg = st.navigation([hw1_page, hw2_page, hw3_page, hw5_page, hw6_page])

pg.run()