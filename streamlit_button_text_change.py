import streamlit as st

with st.form(key="my_form_to_submit"):
    user = st.text_input("Enter some value here ")
    submit_button = st.form_submit_button(label="Submit")
    st.write(user)

# code sample 2 :
import streamlit as st

with st.form(key="my_form_to_submit1"):
    user1 = st.text_input("Enter some value here ")
    submit_button = st.form_submit_button(label="Submit")
    st.write(user1)
if submit_button:
    st.write(user1)

# code sample 3 :

import streamlit as st

result = st.text_input("Enter some value here ")
if st.button("asas"):
    st.write("Hello")
    st.write(result)
