

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation




df = pd.read_csv('bbc-text.csv')

# We keep only the articles

df = df.iloc[:,1]




df.isnull().sum()


# Let's see if there is blank instead of text in the rows



for i in range(0,len(df)):
    if df[i].isspace() is True:
        print("Blank row at row {}".format(i)) 


# No blank row

# Let's start to preprocess our data


cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')


dtm = cv.fit_transform(df)



# # Using LDA



LDA = LatentDirichletAllocation(n_components=5,random_state=42)   #n_components: the number of clusters we are looking for



LDA.fit(dtm)


# # Showing Stored Words



len(cv.get_feature_names())



import random
for i in range(10):
    random_word_id = random.randint(0,len(cv.get_feature_names()))
    print(cv.get_feature_names()[random_word_id])


# # Showing top words by topic

# Let's see how one topic works and look like


len(LDA.components_) # number of topics


single_topic = LDA.components_[0]



# Returns the indices that would sort this array.
single_topic.argsort()

# Word least representative of this topic
single_topic[13214]


# Word most representative of this topic
single_topic[6422]


single_topic.argsort()[-10:]


# Let's see the top 15 per topics


for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


# # Attaching Discovered Topic Labels to Original Articles

topic_results = LDA.transform(dtm)



topic_results[0].argmax()


# This means that our model thinks that the first article belongs to topic #1.



# In[27]:


df = pd.DataFrame(df)



topic_results.argmax(axis=1)


df['Topic'] = topic_results.argmax(axis=1)



# We can determine topics idea from the earlier TOP15 words per topics

# Let's print it again



for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


# It could be like : 
# 
# Topic 0 = People
# 
# Topic 1 = Politics
# 
# Topic 2 = Sports
# 
# Topic 3 = Tech
# 
# Topic 4 = Economics
# 

# Let's add this to our dataset

# In[32]:


df['Topic'] = df['Topic'].apply(lambda x: 'People' if x == 0 else 'Politics' if x == 1 else 'Sports' if x == 2 else 'Tech' if x == 3 else 'Economics')




