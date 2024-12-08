import json
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

intents = json.load(open('intents.json'))
#print(intents)

tags = []
patterns = []

for intent in intents['intents']:
    #print(intent)
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

print(len(tags))  
print(len(patterns))
#output = 405 for both tags and patterns.

vector = TfidfVectorizer()
patterns_scaled = vector.fit_transform(patterns)

Bot = LogisticRegression(max_iter=10000)
Bot.fit(patterns_scaled,tags)

input_message = "Hi"
input_message = vector.transform([input_message])
print(Bot.predict(input_message))
# Output=['greeting']

input_message = "ttyl"
input_message = vector.transform([input_message])
print(Bot.predict(input_message))
# Output = ['goodbye']

def Chatbot(input_message):
    input_message = vector.transform([input_message])
    pred_tag = Bot.predict(input_message)[0]
    for intent in intents['intents']:
        if intent['tag'] == pred_tag:
            response = random.choice(intent['responses'])
            return response
        
# input_message = input('Enter user Message:')
# print(Chatbot(input_message))

st.title("End to End Chatbot using Python")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Chatbot: " + Chatbot((prompt))
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})