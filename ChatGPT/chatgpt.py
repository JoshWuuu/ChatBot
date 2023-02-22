import openai
import streamlit as st
from streamlit_chat import message

def generate_text(prompt, max_tokens=100, temperature=0.9, top_p=1, frequency_penalty=0, presence_penalty=0.6, stop=None):
    """
    generate text with openai api
        
    Input:
    - prompt: str, input sentence
    - max_tokens: int, max length of the output sentence
    - temperature: float, randomness of the sentence, higher temperature, more random
    - top_p: float, randomness of the sentence
    - frequency_penalty: float, randomness of the sentence
    - presence_penalty: float, randomness of the sentence
    - stop: str, stop token

    Returns:
    - response.choices[0].text: str, generated text
    """

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop
    )

    # take the first response
    return response.choices[0].text

def get_text():
    """
    get text from the user
        
    Input:
    - None

    Returns:
    - None
    """
    # get the text from the user
    text = st.text_input("User: ", key="user_input")
    return text
                                  

def main():
    openai.api_key = st.secrets["openai_api_key"]

    st.title("chatBot : Streamlit + openAI")

    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    user_input = get_text()

    if user_input:
        output = generate_text(user_input)
        # store the output
        st.session_state['generated'].append(output)
        # store the input
        st.session_state['past'].append(user_input)
    
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']), -1, -1):
            message("User: ", st.session_state['past'][i], key=str(i) + "_user")
            message("Bot: ", st.session_state['generated'][i], key=str(i) + "_bot")

if __name__ == "__main__":
    main()