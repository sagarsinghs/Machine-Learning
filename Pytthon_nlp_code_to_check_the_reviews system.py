
# coding: utf-8

# In[223]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[224]:


dataset= pd.read_csv('1-restaurant-train.csv',delimiter ='\t',quoting=3)


# In[225]:


import re


# In[226]:


import nltk


# In[227]:


nltk.download('stopwords')


# In[228]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[230]:


review = re.sub('[^a-zA-Z]',' ',dataset['Liked'][0])


# In[196]:


print(review)


# In[231]:


review = review.lower()
review = review.split()


# In[232]:


#object for the porterstemmer class is created and porter stemmer

ps = PorterStemmer()
review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]


# In[233]:


#stem function will remove all the unnecessray words from our review and will only
#keep the necesary words like removed words are the which will simply make the 
#matrix sparsh from the PorterStemmer class and also it will make all the different
#past tense and present or future into just the preent tensse so as to 
# remove more mixtures of same kind if words(eg- loved and love will be just one love)
print(review)


# In[234]:


#joining all the strings words into review as a sting
review =' '.join(review)
print(review)


# In[251]:


def function_review():
    corpus=[]
    for i in range(2,100):
        review = re.sub('[^a-zA-Z]',' ',dataset['Liked'][i])
        review = review.lower()
        review= review.split()
        ps =PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review =' '.join(review)
        corpus.append(review)
    return corpus
       
   


# In[252]:


review_returned= function_review()
print(review_returned)


# In[253]:


#finding the bag of words
from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer()
#it give all those 1500 words whic occurs frreqentlf
cv = CountVectorizer(max_features=80)
X = cv.fit_transform(review_returned).toarray()
np.shape(X)
#1565 are just the unique words appearing in the whole collecion of words


# In[254]:


#max features will remove all the relevant words that are coming less frequently
#trainig of the datasets in such a way that the datas of the reviews ae in the 0th
#position as 0th is Index and last is 0 or 1 at 1st position.
y = dataset.iloc[:,0].values


# In[255]:


print(y)


# In[260]:


from sklearn.cross_validation import train_test_split
X_train,X_Test,y_train,y_test =train_test_split(X,y,test_size=.20,random_state=0)


# In[258]:


#fitting naive bayes to the training data
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

y_pred =classifier.predict(X_Test)
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[259]:


#here 55 has -ve right response and 91 has right +ve response 
#thus out of 200 55+92 predicted correct

