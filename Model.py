#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import pandas as pd
import os
import sent2vec


# In[ ]:


user_df=pd.read_csv('<Enter name of file containing tweets of users>')


# In[ ]:


##Clean up text to remove special characters

# import preprocessor as p
# import re
# import string
# from gensim.parsing.preprocessing import remove_stopwords
# from nltk.tokenize import TweetTokenizer
# tt = TweetTokenizer()
# from string import digits
# import re
# irrelevant_chars="~?!./\:;+=&^%$#@(,)[]_*"
# emoji_pattern = re.compile("["
#         u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                            "]+", flags=re.UNICODE)
# def deep_clean(x):
#     x=x.lower()
#     x=re.sub(r'http\S+', '', x)
#     remove_digits = str.maketrans(' ', ' ', digits)
#     remove_chars = str.maketrans(' ', ' ', irrelevant_chars)
#     x = x.translate(remove_digits)
#     x = x.translate(remove_chars)
#     x = emoji_pattern.sub(r'', x)
#     x=x.replace("-",' ')
#     x=x.replace('!',' ')
#     x=x.replace('?',' ')
#     x=x.replace('@',' ')
#     x=x.replace('&',' ')
#     x=x.replace('$',' ')
#     x=x.replace('``',' ')
#     x=x.replace("'s",' ')
#     x=x.replace("''",' ')
#     return x


# In[ ]:


##NOTE: Students will need to group individual tweets by the user. Here's a snippet:
# user_text={}
# for i in tqdm(range(len(df))): #Here df is the tweet data.
#     user=loc_df['screen_name'].iloc[i]
#     text=loc_df['text'].iloc[i]
#     if user not in user_text:
#         user_text[user]=' '
#     user_text[user]=user_text[user]+' '+text
#user_text_df=pd.DataFrame(user_text.items(),columns=['screen_name','text'])


# In[ ]:


model = sent2vec.Sent2vecModel()
model.load_model('/data/Term-Analysis/twitter_bigrams.bin')

embeddings = model.embed_sentences(user_text_df['cleaned_text'].tolist())
e=embeddings.tolist()

user_text_df['embeddings']=e


# In[ ]:


filename = '/data/Coronavirus-Tweets/Technica23/finalized_model.sav'
model=pickle.load(open(filename, 'rb'))


# In[ ]:


y_pred=model.predict(np.asarray(user_text_df['embeddings'].tolist()))

