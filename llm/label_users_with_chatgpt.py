import time
import openai
import pandas as pd
from dotenv import load_dotenv, find_dotenv   # get the system environment variables
import tiktoken
import os  # get command line arguments

filename = "../data/participant_data_for_subversive_tweets_on_vaccines.csv"
f = open(filename, "r")
userids = pd.read_csv(filename)['userid'].tolist()
texts = pd.read_csv(filename)['text'].tolist()

participants_to_tweets = {}

for indx in range(len(userids)):
    userid = userids[indx]
    text = texts[indx]
    if userid in participants_to_tweets:
        if type(text)!=str:
            continue
        participants_to_tweets[userid] += "\n"+text
    else:
        participants_to_tweets[userid] = text

load_dotenv(find_dotenv())  # load the environment variables
openai.api_key=os.getenv("OPENAI_API_KEY")

messages = []
system_message = "You are deciding whether people are pro-science or anti-science with respect to COVID based on their tweets. Pro-science users generally express opinions that agree with conventional science and anti-science users may propagate conspiracies."
messages.append({"role":"system","content":system_message})

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo") 

# a = 0
with open("chatgpt_classifications.csv", "a+") as f:
    f.write("userid,classification\n")
for user in participants_to_tweets:  # This starts an infinite loop
    messages = []
    system_message = "You are deciding whether people are pro-science or anti-science with respect to COVID based on their tweets. Pro-science users generally express opinions that agree with conventional science and anti-science users may propagate conspiracies."
    messages.append({"role":"system","content":system_message})
    message = "I am trying to classify twitter users as either pro-science or anti-science with respect to COVID. Pro-science users generally express opinions that agree with conventional science and anti-science users may propagate conspiracies. Could you classify a user based on this text. Your output should simply be pro-science, anti-science or unsure: \n"+participants_to_tweets[user]   
    token_count = len(encoding.encode(message))
    if token_count > 4097:
        message_list = message.split()
        current = 1
        
    while token_count > 4097:
        message = ""
        for x in message_list[:-current]:
            message += x + "\n"
    messages.append({"role":"user","content": message})

    response=openai.ChatCompletion.create(
     model="gpt-3.5-turbo",
     messages=messages
    )
    time.sleep(60)

    reply = response["choices"][0]["message"]["content"]
    try:
        with open("chatgpt_classifications.csv", "a+") as f:
            f.write(str(user)+","+reply+"\n")
    except:
        with open("chatgpt_classifications.csv", "a+") as f:
            f.write(str(user)+","+reply+"\n")
#         a += 1
#         if a == 2:
#             break