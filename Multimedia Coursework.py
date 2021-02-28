#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import os
import numpy as np
from nltk.corpus import stopwords

# All of the irrelevent words and punctuation to be removed from document texts
stop_words = stopwords.words('english')
DELIM = '[ \n\t0123456789;:.,&$£/#~@><|!%^*\(\)\"\'-]+'


# In[2]:


# Reads a document with the index docid 

def readfile(path, docid):
    files = sorted(os.listdir(path))
    f = open(os.path.join(path, files[docid]), 'r',encoding='latin-1')
    s = f.read()
    f.close()
    return s


# In[3]:


# Tokenizes each of the words so that words can be easily
# inspected. Text is all converted to lower case too to
# prevent multiple variations of the same word being tokenized

def tokenize(text):
    return re.split(DELIM, text.lower())


# In[4]:


# The values for the terms in all documents are calculated
# so that they can be accessed using an index later with a query

def indextextfiles_RR(path):
    
    N = len(sorted(os.listdir(path)))                      # The number of documents
    postings = {}                                          # The documents which the word appears in
    df = {}                                                # Document Frequency 
    idf_in_all_docs = {}                                   # All IDF values for each of the words  
    tf_in_all_docs = []                                    # Term frequencies for words in all documents 
    list_of_normTF=[]                                      # The list of all documents normalised term frequencies
    final_docCollection = []                               # The final collection of all tfidf values for every term
    uniqueWords = {}                                       # The set used to check whether a query word is in any document
    
    
    for docID in range(N):                                 # Reads each file one by one in the collection of documents
        s = readfile(path, docID)                           
        words = sorted(tokenize(s))                        # Tokenises each of the words in the document
        words = [w for w in words if not w in stop_words]  # Removes all stopwords such as I, a, you... as they are irrelevent 
        specificDocWords = {}                              # Dictionary of all unique words in the document
        normTF = {}                                        # Dictionary of all the normalised term frequencies in a document
            
        for w in words:
            if w!='':
                postings.setdefault(w, set()).add(docID)   # Adds each of the docIDs to a set containng the words          
                
                # Calculates the unique words and their frequencies in individual documents
                if w in specificDocWords.keys():
                    specificDocWords[w] = specificDocWords[w] + 1
                else:
                    specificDocWords[w] = 1 
           
            if w not in uniqueWords.keys():
                uniqueWords[w] = 1 
    
        
        # calculates the normalised tf for each unique word in the document
        words_in_doc_Count = len(words)
        for word in specificDocWords.keys():
            normTF[word] = (specificDocWords[word] / words_in_doc_Count)               
        
        
        # Appends all of the dictionaries with normalised tf into one list
        tf_in_all_docs.append(normTF)    
    
    
    # gets the df values for all terms 
    # -- The number of documents each term appears in 
    # gets the idf values for all terms using the df values
    # -- Log10 ( number of documents / df ) 
    for x in postings.keys():
        df[x] = len(postings[x])
        idf_in_all_docs[x] = np.log10(N / df[x])
        
       
    # tfidf calculated for each of the words in a document by multiplying
    # the idf values of terms by their normalised term frequencies which 
    # are stored in the final document 
    for docID in range(N): 
        list_of_normTF = list(tf_in_all_docs[docID].items())
        tfidf = {}

        for index in range(len(list_of_normTF)):
            ind = list_of_normTF[index]
            word = ind[0]
            tfidf[word] = ind[1] * idf_in_all_docs[word] 

        final_docCollection.append(tfidf)
    
    
    return postings, N, final_docCollection, uniqueWords, specificDocWords, tf_in_all_docs


# In[5]:


def query_RR(postings, qtext):
    words = tokenize(qtext)                                    # Query Tokenised and all stopwords removed as they are 
    words = [w for w in words if not w in stop_words]          # not calculated in the index due to lack of relevence.
    score_of_documents = {}                                    # Dictionary contains docIDs as keys and scores as values
    
    for w in words:
        if w not in uniqueWords.keys():
            return print("The word \'" + w + "\' cannot be found in any of the documents! \nRemove this word from the query and try again.")
        
    allpostings = [postings[w] for w in words if w!='']        # The postings for each individual word in the query

    
    # Loops through all of the postings which the words in 
    # the query can be found in
    for wordDocs in range(len(allpostings)):

        
        # The postings of each individual word in the query
        collectionOfDocs = allpostings[wordDocs]

        
        # Loops through each of the documents which contain the words 
        for docID in collectionOfDocs:
            document = final_docCollection[docID]
            score = 0

            # Looping through each of the words in the query 1 at a time, retrieving
            # the tfidf values of the words
            for queryWords in words:
                if queryWords in document.keys():
                    score += document[queryWords]
                    

            # Each of the query words are added to a dictionary as their keys
            # and their final scores are the values.
            score_of_documents[docID] = score
            
    
    # Orders all of the scores with the highest ranking document appearing first.
    # Only the top 10 relevent documents are outputted for efficiency
    # Reversed so that the highest ranking document is the first displayed 
    final_rankings = {}
    sorted_keys = sorted(score_of_documents, key=score_of_documents.get, reverse=True)

    for w in sorted_keys[:10]:
        final_rankings[w] = score_of_documents[w]
    
    
    return print("The top 10 relevent documents with their scores for the given query are: \n\n" + str(final_rankings))
    #return print("The top 10 relevent documents with their scores for the given query are: \n\n" + str(final_rankings.keys()))


# In[7]:


postings, N, final_docCollection, uniqueWords, specificDocWords, tf_in_all_docs = indextextfiles_RR('docs')
query_RR(postings,'christmas champions mourinho arsenal')


# In[ ]:





# # Testing of the system

# In[8]:


s = readfile('docs', 736)
print(s)
words = sorted(tokenize(s))
words = [w for w in words if not w in stop_words]

print("\nThe number of words in the document when stop words are removed: \n" + str(len(words)))


# In[9]:


print("The raw term frequencies for the last document in the collection: \n\n" + str(specificDocWords))


# In[10]:


print("The normalised versions of the terms in the same document: \n\n" + str(tf_in_all_docs[736]))


# In[11]:


print("The number of documents that contain the specific word \'years\' \n" + str(len(postings['years'])))


# In[12]:


print("All of the documents that contain the specific word \'years\' \n\n" + str(postings['years']) + "\n\nThe final document in the collection, \'document 736\' can be seen in here which is expected")


# In[13]:


print("The idf value of the word \'years\' is: \n"+ str(np.log10(737 / len(postings['years']))))


# In[14]:


print("The tfidf values of the words in the final document: \n"+ str(final_docCollection[736]))


# In[15]:


query_RR(postings,'christmas')
print("\nThe document \'736\' is the most relevent for the word \'christmas\'. Therefore, one final test should show that the addition of both the scores of \'years\' and \'christmas\' should be equal to 0.093514730898513. It passes!")


# In[16]:


query_RR(postings,'christmas years')


# In[19]:


query_RR(postings,'christmas years nonExistantWORD')
print("\nThe system is also robust to words which cannot be found in any of the documents!")

