# Streamlit Version
# tutorial: http://www.geeksforgeeks.org/a-beginners-guide-to-streamlit/
# uses chatgpi to create instructions sets/code
# ref https://medium.com/@avra42/how-to-build-a-chatbot-with-chatgpt-api-and-a-
# conversational-memory-in-python-8d856cda4542

# CONSTANTS
APP_ID_NAME = "CHAT INTERFACE ALKEMIE TECHNOLOGIES - Personal Assistant"


import os
import openai
# import dotenv # pip install python-dotenv moved to fn

import streamlit as st
##from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
# from langchain.llms import OpenAI

# line changed due to requirement to import ChatOpenAI vs OpenAI
from langchain.chat_models import ChatOpenAI as OpenAI

def get_streamlight_open_api_key():
    # This code was initially used for CLI version and is reused here for expediency - tech debt TODO
    from dotenv import load_dotenv
    # Load the API key from the .env file
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    print("get_streamlight_open_api_key", api_key)
    return api_key

st.set_page_config(page_title='🧠MemoryBot🤖', layout='wide')
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# METHODS: Front End (Streamlight)
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()

def get_text():
    """
    Get the user input text.
    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your AI assistant here! Ask me anything ...",
                            label_visibility='hidden')
    return input_text

with st.sidebar.expander(" 🛠️ Settings ", expanded=False):
    # Option to preview memory store
    if st.checkbox("Preview memory store"):
        st.write(st.session_state.entity_memory.store)
    # Option to preview memory buffer
    if st.checkbox("Preview memory buffer"):
        st.write(st.session_state.entity_memory.buffer)
    MODEL = st.selectbox(label='Model',
                         options=['gpt-3.5-turbo'])
    # options = ['gpt-3.5-turbo', 'text-davinci-003', 'code-davinci-002']) # original code but replaced by above since remaining options threw error
    TEMPERATURE = st.selectbox(label='Temperature',
                         options=[0.5, 0, 1])
    K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)

# Set up the Streamlit app layout
st.title("🔍 Generative Chatbot 🧐")  # https://unicode.org/emoji/charts/full-emoji-list.html
st.markdown(
        ''' 
        > :black[**A Chatbot that remembers,**  *powered by -  [LangChain]('https://langchain.readthedocs.io/en/latest/modules/memory.html#memory') + 
        [OpenAI]('https://platform.openai.com/docs/models/gpt-3-5') + 
        [Streamlit]('https://streamlit.io') + [DataButton](https://www.databutton.io/)*]
        ''')
# st.markdown(" > Powered by -  🦜 LangChain + OpenAI + Dotenv + Streamlit")

# Ask the user to enter their OpenAI API key

key = get_streamlight_open_api_key() # currently persisted in .env file
#print(key)

API_O = st.sidebar.text_input(":blue[Enter Your OPENAI API-KEY :]",
                placeholder="Paste your OpenAI API key here (sk-...)",
                type="password") # Session state storage would be ideal
# print("API_O", API_O)

if len(API_O) == 0:
    API_O = key

if API_O:
    # Create an OpenAI instance
    llm = OpenAI(temperature=TEMPERATURE,
                 openai_api_key=API_O,
                 model_name=MODEL,
                 verbose=False)

    # Create a ConversationEntityMemory object if not already created
    if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K)

    # Create the ConversationChain object with the specified configuration
    Conversation = ConversationChain(
        llm=llm,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=st.session_state.entity_memory
    )
else:
    st.markdown(''' 
        ```
        - 1. Enter API Key + Hit enter 🔐 

        - 2. Ask anything via the text input widget

        Your API-key is not stored in any form by this app. However, for transparency ensure to delete your API once used.
        ```

        ''')
    st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
    # st.sidebar.info("Your API-key is not stored in any form by this app. However, for transparency ensure to delete your API once used.")

# Implementing a Button to Clear the memory and calling the new_chat() function
st.sidebar.button("New Chat", on_click=new_chat, type='primary')

# Get the user INPUT and RUN the chain.
# Also, store them — that can be dumped in the future in a chat conversation format
user_input = get_text()
if user_input:
    output = Conversation.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        st.info(st.session_state["past"][i], icon="🧐")
        st.success(st.session_state["generated"][i], icon="🧠")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download', download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session

import streamlit as st
import pandas as pd
from io import StringIO

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

'''
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
'''

st.caption('William Collins - All Rights Reserved')

quit()

#####################################################################################


#####################################################################################


# METHODS: Back End

# Generate a response
def generate(prompt, model_engine:"text-davinci-003", temperature=0):
    get_open_api_key()
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        stop=None,
        temperature=temperature,
        max_tokens=1024,
        n=1,
    )
    response = completion.choices[0].text
    print(response)
    return response

def get_open_api_key():
    from dotenv import load_dotenv
    # Load the API key from the .env file
    load_dotenv()
    api_key = os.getenv('API_KEY')
    print(OPENAI_API_KEY)


# Defining main function
def main():

    # Define OpenAI API key
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception as e:
        print
        e.message, e.args
        quit("quitting due to API Key failure..")

    '''
    create a loop that:
     1. s the user to read the first response
     2. add an option to provide additional Engineering Prompts
     3. regenerate the response based up that Prompt and print it
     4. Save old and new versions to variables and label/concat content if needed
     4. go back to step 2

    '''

    # Set up the model and prompt
    model_engine = "text-davinci-003"
    prompt = "Create an article about the next great improvements in the future of Artificial Intelligence"
    temperature = 0

    l = 0
    while l < 10:
        print(2)
        print(APP_ID_NAME)
        print("Default prompt:", prompt)
        p = input("Create Engineering Prompt:  ")  # get user input for prompt e.g. chatgpt
        if len(p) < 1:
            print("using default: ", prompt)
        else:
            prompt = p  # use the users input
        l = +1

        print("Processing...  ", prompt)
        result = generate(prompt, model_engine, temperature)
        print("version: ", str(l))


quit() # disable for now since this file/version is a Streamlight Integration

if __name__ == "__main__":
    main()


