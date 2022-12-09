# pip install -r /path/to/requirements.txt

import streamlit as st
#import SessionState
import pandas as pd
import joblib 
#import plotly.express as px

#from tensorflow.keras.models import load_model
import numpy as np
import pickle

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from gensim.models.word2vec import Word2Vec



import bz2
import pickle
import _pickle as cPickle

# Load any compressed pickle file
def decompress_pickle(file):
 data = bz2.BZ2File(file, 'rb')
 data = cPickle.load(data)
 return data



# reading in dataset
df = pd.read_csv("data/tok_lem_sentence.csv")
#st.write(df.head())



# Title
st.header("Get Movie Recommendations from an Auto Complete Thoughts Prediction App")

text=st.text_input("What's on your mind?")


def Predict_Next_Words(model, tokenizer, text):
    for i in range(3):
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = np.array(sequence)
        
        preds = loaded_model.predict(sequence)
        pp=(-preds[0]).argsort()[:10]
        #pp=pp[0:5]
        #preds=np.argmax(preds,axis=1)
#         print(preds)
        predicted_word = []
        
        for key, value in tokenizer.word_index.items():
               for word in pp:   
                       if value == word:
                        predicted_word.append(key)
                        break
        
        #print(predicted_word)
        return predicted_word



def is_word_in_model(word, model):
    """
    Check on individual words ``word`` that it exists in ``model``.
    """
    assert type(model).__name__ == 'KeyedVectors'
    is_in_vocab = word in model.key_to_index.keys()
    return is_in_vocab

def predict_w2v(query_sentence, dataset, model, topk=3):
    query_sentence = query_sentence.split()
    in_vocab_list, best_index = [], [0]*topk
    for w in query_sentence:
        # remove unseen words from query sentence
        if is_word_in_model(w, model.wv):
            in_vocab_list.append(w)
    # Retrieve the similarity between two words as a distance
    if len(in_vocab_list) > 0:
        sim_mat = np.zeros(len(dataset))  # TO DO
        for i, data_sentence in enumerate(dataset):
            if data_sentence:
                sim_sentence = model.wv.n_similarity(
                        in_vocab_list, data_sentence)
            else:
                sim_sentence = 0
            sim_mat[i] = np.array(sim_sentence)
        # Take the five highest norm
        best_index = np.argsort(sim_mat)[::-1][:topk]
    return best_index

def extract_best_indices(m, topk, mask=None):
    """
    Use sum of the cosine distance over all tokens.
    m (np.array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
    topk (int): number of indices to return (from high to lowest in order)
    """
    # return the sum on all tokens of cosinus for each sentence
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0) 
    else: 
        cos_sim = m
    index = np.argsort(cos_sim)[::-1] # from highest idx to smallest score 
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask) #eliminate 0 cosine distance
    best_index = index[mask][:topk]  
    return best_index


def predict_spacy(model, query_sentence, embed_mat, topk=5):
    """
    Predict the topk sentences after applying spacy model.
    """
    query_embed = model(query_sentence)
    mat = np.array([query_embed.similarity(line) for line in embed_mat])
    # keep if vector has a norm
    mat_mask = np.array(
        [True if line.vector_norm else False for line in embed_mat])
    best_index = extract_best_indices(mat, topk=topk, mask=mat_mask)
    return best_index


def get_recommendations_tfidf(sentence, tfidf_mat,tokenizer):
    
    """
    Return the database sentences in order of highest cosine similarity relatively to each 
    token of the target sentence. 
    """
    # Embed the query sentence
    tokens = [str(tok) for tok in tokenizer(sentence)]
    vec = vectorizer.transform(tokens)
    # Create list with similarity between query and dataset
    mat = cosine_similarity(vec, tfidf_mat)
    # Best cosine distance for each token independantly
    print(mat.shape)
    best_index = extract_best_indices(mat, topk=3)
    return best_index


print_sentence1 = st.empty()

print_sentence2 = st.empty()

#ss = SessionState.get(button1 = False)

ques1 = st.radio(

    "Click yes to complete sentence!",

    ('No','Yes'))


if ques1 == "Yes":
    # Load the model and tokenizer
    tokenizer = pickle.load(open('model/tokenizer_avi.pkl', 'rb'))
    # load json and create model
    json_file = open('model/nextword.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model/nextword_avi.h5")

    #text= 'adventures with' 
    text2 = text.split(" ")
    text2= text2[-1]

    text2 = ''.join(text2)
    predicted_words = Predict_Next_Words(loaded_model, tokenizer, text)
    predicted_sentence = " ".join(predicted_words)
    mk_sentnce = f'<p style="font-family:Courier; color:Blue; font-size: 20px;">{text} "{predicted_sentence}"</p>'
    print(predicted_words)
    st.markdown(f"Predicted Sentence:")
    st.write(f"{text} '*{predicted_sentence}* '")
    #st.markdown(mk_sentnce,unsafe_allow_html=True)
    



st.subheader("Press the button below to get movie recommendations based on this sentence")

# If button is pressed
if st.button("Get movie recommendations"):

    #st.write(f"Movie Recommendation: Frankie Starlight, The quirky story of a young boy's adventures growing up with his stunningly beautiful mother and the two very different men who love her.")
    # Create model
    # word2vec_model = Word2Vec(min_count=0, workers = 8, vector_size=275) 
    # # Prepare vocab
    # word2vec_model.build_vocab(df.tok_lem_sentence.values)
    # # Train
    # word2vec_model.train(df.tok_lem_sentence.values, total_examples=word2vec_model.corpus_count, epochs=30)
    # Predict
    with st.spinner(text="In progress"):
        #Create model
        word2vec_model = Word2Vec(min_count=0, workers = 8, vector_size=275) 
        # Prepare vocab
        word2vec_model.build_vocab(df.tok_lem_sentence.values)
        # Train
        word2vec_model.train(df.tok_lem_sentence.values, total_examples=word2vec_model.corpus_count, epochs=10)
        # # word2vec_model = decompress_pickle("../model/word2vec_model_avi.pbz2") 
        # with open("../model/vectorizer_avi.pkl", "rb") as f:
        #     vectorizer = pickle.load(f)
        query_sentence = f"{text} {predicted_sentence}"
        best_index = predict_w2v(query_sentence, df['tok_lem_sentence'].values, word2vec_model)    
        final_output = df[['original_title', 'genres', 'sentence']].iloc[best_index]
        st.markdown("Recommended movies:")
        st.write(final_output)



        
        # Predict
        # df['spacy_sentence'] = df['spacy_sentence'].apply(lambda x: x.split(" "))
        # embed_mat = df['spacy_sentence'].values
        # with open("../model/spacy_avi_embed.pkl", "rb") as f:
        #     embed_mat = pickle.load(f)
        # #print(embed_mat)
        # #st.write(embed_mat[0:2])
        # best_index = predict_spacy(nlp, query_sentence, embed_mat)
        # final_output = df[['original_title', 'genres', 'sentence']].iloc[best_index]
        # st.write(final_output)


        # vecotrizer output
        # tfidf_mat = vectorizer.fit_transform(df['sentence'].values) # -> (num_sentences, num_vocabulary)
        # best_index = get_recommendations_tfidf(query_sentence, tfidf_mat)
        # final_output = df[['original_title', 'genres', 'sentence']].iloc[best_index]
        # st.write(final_output)


        

