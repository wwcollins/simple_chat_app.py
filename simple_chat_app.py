
# uses chatgpi to create instructions sets/code
# ref https://medium.com/@avra42/how-to-build-a-chatbot-with-chatgpt-api-and-a-
# conversational-memory-in-python-8d856cda4542

import openai
import os
import dotenv # pip install python-dotenv


# CONSTANTS
APP_ID_NAME = "CHAT INTERFACE ALKEMIE TECHNOLOGIES - Personal Assistant"
OPENAI_API_KEY = "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"


# METHODS

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


if __name__ == "__main__":
    main()


