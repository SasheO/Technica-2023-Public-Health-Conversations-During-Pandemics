import openai
import pandas as pd
import time

from dotenv import load_dotenv, find_dotenv   # get the system environment variables

load_dotenv('.env')
import os  # get command line arguments

# Load example data
pro_science_df = pd.read_csv('/Users/saradeshmukh/Documents/Technica2023PublicHealthConversations/Technica-2023-Public-Health-Conversations-During-Pandemics/data/health_experts50.csv')
anti_science_df = pd.read_csv('/Users/saradeshmukh/Documents/Technica2023PublicHealthConversations/Technica-2023-Public-Health-Conversations-During-Pandemics/data/pseudo_health_experts50.csv')

df = pd.read_csv('/Users/saradeshmukh/Documents/Technica2023PublicHealthConversations/Technica-2023-Public-Health-Conversations-During-Pandemics/data/participant_data_for_subversive_tweets_on_vaccines.csv')
tweets = df['text'].tolist() # reads the "text" or the tweets

load_dotenv(find_dotenv())  # load the environment variables
openai.api_key=os.getenv("OPENAI_API_KEY")

messages = []
results = []
system_message = "You are deciding whether people are pro-science or anti-science based on their tweets."
#messages.append({"role":"system","content":system_message})

#print("Alright! I am ready to be your advisor chatbot" + "\n")

for tweet in tweets:

    message = f"You are deciding whether people are pro-science or anti-science based on their tweets."
    messages.append({"role": "user", "content": message})

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )

    # Access the generated text using the correct attribute
    reply = response['choices'][0]['message']['content'].strip()

    # if "pro-science" in reply:
    #     classification = "pro-science"
    # elif "anti-science" in reply:
    #     classification = "anti-science"
    # else:
    #     classification = "neutral"

    # results.append({
    #     'tweet' : tweet,
    #    'label': classification
    # })

    print("Tweet: " + tweet + " - " + reply + "\n")

#print(results)

# results_df = pd.DataFrame(results)

# results_df.to_csv('participant_results.csv', index=False)
    
    
print("Bye!!!")
