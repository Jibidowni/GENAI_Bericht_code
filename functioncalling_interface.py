from openai import OpenAI
import streamlit as st
import json

client = OpenAI(api_key="")

# Streamlit UI
st.title("Chat with OpenAI")
st.write("A simple chat interface using OpenAI's API")

# User input
user_input = st.text_input("You:", value="", key="input")

# Function
def middle_letter(word: str):
    length = len(word)
    if length % 2 == 1:
        middle = length // 2 
        return length, word[middle]
    else:
        middle = word[length // 2-1 :length // 2+1]
        return length, middle

# Chat history
system_prompt = "be helpful"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# Tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "middle_letter",
            "description": "Returns the middle letter of a given word. Call this whenever the user wants to know the middle letter of a word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "The word in question.",
                    },
                },
                "required": ["word"],
                "additionalProperties": False,
            },
        }
    }
]

# Generate response
if st.button("Send"):
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.messages,
            tools=tools,
            temperature=1,
            top_p=1
        )
        
        # Check if the assistant wants to call a function
        if response.choices[0].finish_reason == "tool_calls":
            # Extract function call details
            parameters_json_str = response.choices[0].message.tool_calls[0].function.arguments
            parameters = json.loads(parameters_json_str)
            word = parameters["word"]
            
            # Execute the function
            length, middleLetter = middle_letter(word=word)
            if len(middleLetter) == 1:
                function_result = f"Das Wort '{word}' hat {length} Buchstaben. Der mittlere Buchstabe ist der {length // 2 +1}. Buchstabe: {middleLetter}"
            else:
                function_result = f"Das Wort '{word}' hat {length} Buchstaben. Somit gibt es 2 mittlere Buchstaben, und zwar der {length // 2}. und der {length // 2 +1}. Buchstaben: {middleLetter}"
            
            # Append the function result to the messages
            st.session_state.messages.append({
                "role": "assistant",
                "name": response.choices[0].message.tool_calls[0].function.name,
                "content": function_result
            })
        else:
            # Assistant's reply is already in msg
            final_reply = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": final_reply})

    # Display chat history
    for message in st.session_state.messages:
        # Skip function call messages and assistant messages with content None
        if message["role"] == "function":
            continue
        if message["role"] == "assistant" and message.get("content") is None:
            continue
        if message["role"] == "user":
            st.text(f"You: {message['content']}")
        elif message["role"] == "assistant":
            st.write("AI:")
            st.text_area(label="", value=message['content'], height=100, disabled=True)
        # Optionally, handle system messages or other roles if needed
