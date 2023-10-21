import openai
from dotenv import load_dotenv, find_dotenv   # get the system environment variables
import os  # get command line arguments

load_dotenv(find_dotenv())  # load the environment variables
openai.api_key=os.getenv("OPENAI_API_KEY")

messages = []
system_message = "You are deciding whether people are pro-science or anti-science based on their tweets"
messages.append({"role":"system","content":system_message})

print("Alright! I am ready to be your advisor chatbot" + "\n")

while True:  # This starts an infinite loop
    user_input = input("How Can I assist: (type 'exit' to quit): ")
    if user_input.lower().strip() == "exit":  # Convert the input to lowercase for a case-insensitive comparison
        break  # This will exit the loop if "exit" is typed

    message = user_input    
    messages.append({"role":"user","content": message})

    response=openai.ChatCompletion.create(
     model="gpt-4",
     messages=messages
    )

    reply = response["choices"][0]["message"]["content"]
    print(reply)
print("Bye!!!")
