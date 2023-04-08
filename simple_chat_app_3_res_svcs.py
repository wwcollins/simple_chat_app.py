# Streamlit Version
# tutorial: http://www.geeksforgeeks.org/a-beginners-guide-to-streamlit/
# uses chatgpi to create instructions sets/code
# ref https://medium.com/@avra42/how-to-build-a-chatbot-with-chatgpt-api-and-a-
# conversational-memory-in-python-8d856cda4542

# CONSTANTS
APP_ID_NAME = "CHAT INTERFACE ALKEMIE TECHNOLOGIES - Personal Assistant for Job Seekers"

import os
import os.path
import openai
# import dotenv # pip install python-dotenv moved to fn # see def

# HTML Components test and embed iframe
import streamlit.components.v1 as components  # Import Streamlit


##from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
# from langchain.llms import OpenAI

import streamlit as st
import pandas as pd
from io import StringIO

# line changed due to requirement to import ChatOpenAI vs OpenAI
from langchain.chat_models import ChatOpenAI as OpenAI

import requests
from dotenv import load_dotenv

import streamlit_authenticator as stauth
import yaml  # pip install pyaml
from yaml.loader import SafeLoader

import time
import streamlit as st

# METHODS
def get_github_version():
    # url example: https://api.github.com/repos/{owner}/{repo}/releases/latest
    url = "https://api.github.com/repos/wwcollins/simple_chat_app.py/releases/latest"
    try:
        response = requests.get("https://api.github.com/repos/v2ray/v2ray-core/releases/latest")
        version_name = response.json()["name"]
        print("github version name", version_name)
        return version_name
    except Exception as e:
        print("An Error occurred trying to pull version number from Github:", e)

def get_streamlight_open_api_key():
    # This code was initially used for CLI version and is reused here for expediency
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
if "input" not in st.session_state:  # Error thrown on cloud side - TODO Debug and Fix
    try:
        st.session_state["input"] = ""
    except Exception as e:
        st.warning("An error occured while initiating session state. Ensure your key is entered correctly")
        print(e)
        st.session_state.entity_memory.store = {}
        st.session_state.entity_memory.buffer.clear()
        st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# STREAMLIT COMPONENTS FOR HTML - Test
app_path = "https://wwcollins-simple-chat-app-py-simple-chat-app-3-res-svcs-bc5wk7.streamlit.app" # did not work, rendered poorly
# Render the h1 block, contained in a frame of size 200x200.
# components.html("<html><body><h1><iframe src='http://" + app_path  + "/' height='100%' width='100%'></iframe>'</h1></body></html>', width=200, height=200")


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
    # st.text_input(label, value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, *, placeholder=None, disabled=False, label_visibility="visible")
    input_text = st.text_input("You: ", st.session_state["input"], max_chars=10000,  key="input", placeholder="Your AI assistant here! Ask me anything ...")
    if len(input_text) > 4000:
        # st.balloons()
        st.write('You have exceeded the suggested character limit of 4000')
    return input_text


# END METHODS - main code

    # Build Side Bar

with st.sidebar.expander(" 🛠️ Settings ", expanded=False): # TODO - leverage this code e.g. Resume expansion of sections
    # Option to preview memory store
    if st.checkbox("Preview memory store"):
        st.write(st.session_state.entity_memory.store)
    # Option to preview memory buffer
    if st.checkbox("Preview memory buffer"):
        st.write(st.session_state.entity_memory.buffer)
    MODEL = st.selectbox(label='Model',
                         options=['gpt-3.5-turbo', 'text-davinci-0o03', 'code-davinci-0o02'])
    # options = ['gpt-3.5-turbo', 'text-davinci-003', 'code-davinci-002']) # original code but replaced by above since remaining options threw error
    TEMPERATURE = st.selectbox(label='Temperature',
                         options=[0.5, 0, 1])
    K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)

# Set up the Streamlit app layout

# Authenticator (streamlit - authenticator)
authenticate_app = False # TODO address issues when this is set to True
if authenticate_app:
    # Fixed Error thrown below, check if file exists
    path = 'compose-dev.yaml'
    bcheck_file = os.path.isfile(path)
    print(bcheck_file, path)

    if bcheck_file:
        with open(path, 'r') as file:
            config = yaml.load(file, Loader=SafeLoader)
            # print("config credentials = " + config['cookie']['name'])
            # print("config credentials = " + config['cookie']['key'])
            print("config = ", config['credentials']['usernames'])
            st.write ("config = ", config['credentials']['usernames'])
            # st.write("config email = ", config['credentials']['usernames']['email'])
            st.write("config expiry days cookie = ", config['cookie']['expiry_days'])
            st.write('config preauth emails = ', config['preauthorized']['emails'])
            usernames_list = []
            for x in config['credentials']['usernames']:
                usernames_list.append(x)
                st.write(x)
                st.write(usernames_list)
            st.write('construct pw list')
            password_list = []

            # print(people[1]['name'])
            for y in config['credentials']['usernames']:
                st.write(y) # usernames printed
                st.write(y['password'].value())
                st.write(y.value()[0])
                # st.write(y[0]['password'].item)


                if y[0]['password'] == 'password':
                    password_list.append(y[0]['password'][0])
                    st.write(y[0]['password'][0])
                    st.write(password_list)
    else:
        print(path + " not found...")

    # example: authenticator = stauth.Authenticate(names, usernames, hashed_passwords, 'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)

    hashed_passwords = stauth.Hasher(['!@Connor63', '!@Connor62']).generate()
    print (hashed_passwords)

    authenticator = stauth.Authenticate(
        #config['credentials'],
        usernames_list,
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']['emails']
    )

    # for usernames, names, passwords in zip(usernames, names, passwords):
    #    user_dict = {"name": usernames, "password": passwords}
    #    credentials["usernames"].update({usernames: user_dict})

    # authenticator = stauth.Authenticate("credentials", "cookie_name", "random_key", cookie_expiry_days=30)

    # authenticator = stauth.Authenticate(names, usernames, hashed_passwords, 'alk_ai_assistant', 'hellodolly', cookie_expiry_days=30)


    # Render the login widget by providing a name for the form and its location
    # (i.e., sidebar or main):
    name, authentication_status, username = authenticator.login('Login', 'main')
    print("User and Credential Auth Status: " + str(authentication_status))
    st.write("User and Credential Auth Status: " + str(authentication_status))
    # print("User and Credential Info: " + name + authentication_status + username)
    print("User and Credential Info: " + str(name))
    st.caption("User and Credential Info: " + str(name))


    if st.session_state["authentication_status"]:
        authenticator.logout('Logout', 'main')
        st.write(f'Welcome *{st.session_state["name"]}*')
        # st.title('Some content')
    elif st.session_state["authentication_status"] == False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] == None:
        st.warning('Please enter your username and password')



    # end authenticator

st.title("🔍 Generative AI Assistant 🧐")
try:
    gh_version = get_github_version()
    st.caption(gh_version + " with authenticator")
except Exception as e:
    st.caption("version not currently available" + " with authenticator")

# https://unicode.org/emoji/charts/full-emoji-list.html
st.markdown(
        ''' 
        > :black[**A Context-Based Generative AI Bot with Authenticator,  *powered by -  [LangChain]('https://langchain.readthedocs.io/en/latest/modules/memory.html#memory') + 
        [OpenAI]('https://platform.openai.com/docs/models/gpt-3-5') + 
        [Streamlit]('https://streamlit.io') + [DataButton](https://www.databutton.io/)*]
        ''')
# st.markdown(" > Powered by -  🦜 LangChain + OpenAI + Streamlit")



# with st.expander('more...'):  # TODO this is throwing exception re nested expanders not allowed TODO investigate!
    #with st.expander("Resume Information", expanded=False):
        #st.write("...additional info here...")




# -- start demo buttons
# TODO create demo buttons that pre-populates the main input box - In Progress
# -- end demo buttons


# Ask the user to enter their OpenAI API key
key = get_streamlight_open_api_key() # currently persisted in .env file
if key == 0:
    st.warning("Your API Key is not entered properly or not the correct length. you can get a key of your own at https://platform.openai.com/account/api-keys.")
#print(key)

# st.text_input(label, value="", max_chars=None, key=None, type="default", help=None,
# autocomplete=None, on_change=None, args=None, kwargs=None, *, placeholder=None, disabled=False,
# label_visibility="visible")
openai_apikey_help = "After you have signed up for an OpenAi account you can acquire a " \
                     "key of your own at https://platform.openai.com/account/api-keys.  Simply" \
                     "Paste it into this box and press enter.  The key will be good for that session." \
                     "If you would like to contact us for help please do so at techsupport@williamwcollinsjr.com"
API_O = st.sidebar.text_input(":blue[Enter Your OPENAI API-KEY :]",value=key,
                placeholder="Paste your OpenAI API key here (sk-...)",
                type="password", help="must be present to work!") # Session state storage would be ideal
# print("API_O", API_O)

if len(API_O) < 10:
    API_O = 0


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

########### GET USER INPUT AND PROCESS ###############
# Get the user INPUT and RUN the chain.
# Also, store them — that can be dumped in the future in a chat conversation format
user_input = get_text()
#  TODO Opportunity to analyze returned text from user here - bad stuff, formatting, warnings, etc.  Use
# corresponding python libs NLP etc to do this

len_user_input = str(len(user_input))
# st.caption ("Chars = " + len_user_input)


if user_input:

    with st.spinner("processing your request...  this might take awhile"):
        time.sleep(5)
        try:
            output = Conversation.run(input=user_input)
            output_len = len(output)
            st.caption ("Assistant output length: ", output_len)
            if len(output) == 0:
                st.warning("We apologize.  The AI Assistant Engine returned no response. Try rerunning your"
                           " request.  If this does not help please clear your browser cache.")
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
        except Exception as e:
            print ('Error:', e)

        st.success('Done! See response, below')
        st.balloons()

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        st.info(st.session_state["past"][i], icon="🧐")
        st.success(st.session_state["generated"][i], icon="🧠")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    # TODO Can throw error - may require fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download', download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        # with st.sidebar.expander(label= f"Conversation-Session:{i}"):
        with st.sidebar.expander(label=f"Conversation:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session

uploaded_file = st.file_uploader("Choose a file to upload")

if uploaded_file is not None:
    print('Read and write file as bytes')
    # bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    print('convert to a string based IO')
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    print('To read file as string')
    string_data = stringio.read()
    st.write(string_data)

    # Follwing Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

st.caption('William Collins - All Rights Reserved')

quit()

#####################################################################################


#####################################################################################


# METHODS: Back End

# Generate a response
def generate(prompt, model_engine: "text-davinci-0o03", temperature=0):
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
    print(api_key)


# Defining main function
def main():

    # Define OpenAI API key
    try:
        openai.api_key = get_open_api_key()
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
    model_engine = "text-davinci-0o03"
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


