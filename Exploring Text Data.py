#!/usr/bin/env python
# coding: utf-8

# ## Exploring Text Data
# 
# Working with text is generally more challenging than working with numerical data. Hence, any kind of technique that helps in generating an intuition of the existing dataset is welcome. One of the simplest approach to understand any text document or to compare multiple documents can be to compute a frequency table of individual words present in the document/documents and use it to conduct further experiements like: finding top words per document, finding top common words among documents etc.
# 
# In our case, we have taken the challenge of Analyzing Sentiments from Twitter data, so we will focus on how to generate word frequencies and use it to create **Word Clouds** in Python that will help us get a better overall understanding of the dataset.
# 
# 
# ### Table of Contents
# 1. About the Dataset
# 2. Generating Word Frequency
# 3. EDA using Word Clouds
# 4. Why to Preprocess text data?
# 5. Challenge

# ### 1. About the Dataset
# 
# The dataset that we are going to use is the same dataset of tweets from Twitter that will be used in module 8 for **Social Media Information Extraction**. You can download it from [here.](https://studio.trainings.analyticsvidhya.com/assets/courseware/v1/aa0ae6514e0be95f11be85b84d4fd6d2/asset-v1:AnalyticsVidhya+NLP101+2018_T1+type@asset+block/tweets.csv)
# Let's load the dataset using pandas and have a quick look at some sample tweets. 

# In[2]:


#Load the dataset
import pandas as pd 
dataset = pd.read_csv('tweets.csv', encoding = 'ISO-8859-1')

dataset.head()


# As can be seen above, **text** column is of interest to us as it contains the tweet. At this point, you don't have to worry about other columns as that will be handled in future modules. Let's go ahead and inspect some of the tweets.
# 
# ### 2. Generating Word Frequency
# 
# Let's first generate a frequency table of all the words present in all the tweets combined.

# In[3]:


def gen_freq(text):
    
    word_list = []
    for tw_words in text.split():
        word_list.extend(tw_words)

    word_freq = pd.Series(word_list).value_counts()
    word_freq[:20]
    
    return word_freq

gen_freq(dataset.text.str)


# ### 3. EDA using Word Clouds
# 
# Now that you have succesfully created a frequency table, you can use that to create multiple **visualizations** in the form of word clouds. Sometimes, the quickest way to understand the context of the text data is using a word cloud of top 100-200 words. Let's see how to create that in Python.
# 
# **Note:-** You'll use the `WordCloud` library of Python. You can install it by - 
# 
# `pip install wordcloud`

# In[5]:


#Import libraries
import matplotlib.pyplot as plt
from wordcloud import WordCloud

word_freq = gen_freq(dataset.text.str).iloc[:100]
wc = WordCloud(width=400, height=330, max_words=100, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# **Few things to Note:-**
# 
# 1. There is noise in the form of "RT" and "&amp" which can be removed from the word frequency.
# 2. Stop words like "the", "in", "to", "of" etc. are obviously ranking among the top frequency words but these are just constructs of the English language and are not specific to the people's tweets.
# 3. Words like "demonetization" have occured multiple times. The reason for this is that the current text is not **Normalized** so words like "demonetization", "Demonetization" etc. are all considered as different words.
# 
# The above are some of the problems that we need to address in order to make better visualization. Let's solve some of the problems!
# 
# #### Text Cleaning
# 
# You have already learnt how to utilize Regex to do text cleaning, that is precisely what we are doing here.

# In[33]:


import re

def clean_text(text):
   
    text = re.sub(r'RT', '', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'^[#@<>!$]', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    text = text.lower()
    return text


# The above will solve problems related to RT, &amp and also the problem of counting same word twice due to case difference. Yet we can do better, let's remove the common stop words.
# 
# #### Stop words Removal
# WordCloud provides its own stopwords list. You can have a look at it by- 
# 

# In[34]:


from wordcloud import STOPWORDS

print(STOPWORDS)


# Now that you know what all has to be changed to improve our word cloud, let's make some wordclouds. We'll call the previous functions of `clean_text()` and `gen_freq()` to perform cleaning and frequency computation operation respectively and drop the words present in `STOPWORDS` from the `word_freq` dictionary.

# In[35]:


text = dataset.text.apply(lambda x: clean_text(x))
word_freq = gen_freq(text.str)*100
word_freq = word_freq.drop(labels=STOPWORDS, errors='ignore')

wc = WordCloud(width=450, height=330, max_words=200, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(8,8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# Now that you have succesfully created a wordcloud, you can get some insight into the areas of interest of the general twitter users:
# 
#  - It is evident that people are talking about govt. policies like **demonetization**, **J&K**. 
#  - There are some personalitites that are mentioned numerous times like **evanspiegel**, **PM Narendra Modi**, **Dr Kumar Vishwas** etc.
#  - There are also talks about **oscars**, **youtube** and **terrorists**
#  - There are many sub-topics that revolve around demonetization like **atms**, **bank**, **cash**, **paytm** etc. Which tells that many people are concerned about it.
#  
# 
# 
# Also something to note is even now some words are misreperesented for example: **modi**, **narendra** and **narendramodi** all refer to the same person. This can eaisly be solved by **Normalizing** 

# In[ ]:




