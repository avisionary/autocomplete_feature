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


print_sentence1 = st.empty()

print_sentence2 = st.empty()

#ss = SessionState.get(button1 = False)

ques1 = st.radio(

    "Ready to get prediction?",

    ('No','Yes'))


if ques1 == "Yes":
    # Load the model and tokenizer
    tokenizer = pickle.load(open('../model/tokenizer_avi.pkl', 'rb'))
    # load json and create model
    json_file = open('../model/nextword.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("../model/nextword_avi.h5")

    #text= 'adventures with' 
    text2 = text.split(" ")
    text2= text2[-1]

    text2 = ''.join(text2)
    predicted_words = Predict_Next_Words(loaded_model, tokenizer, text)
    predicted_sentence = " ".join(predicted_words)
    mk_sentnce = f'<p style="font-family:Courier; color:Black; font-size: 15px;">Predicted Sentence:</p><p style="font-family:Courier; color:Blue; font-size: 20px;">{text} "{predicted_sentence}"</p>'
    print(predicted_words)
    #st.markdown(f"Predicted Sentence:")
    st.markdown(mk_sentnce,unsafe_allow_html=True)
    

    picked_word=st.text_input("Which predicted word would like to go forward with?")

    if picked_word=='adventures':
        st.write(f"Movie Recommendation: Frankie Starlight, The quirky story of a young boy's adventures growing up with his stunningly beautiful mother and the two very different men who love her.")
        














        # print(predicted_word)









    # # Unpickle classifier
    # clf = joblib.load("StackedPickle.pkl")
    
    # # Store inputs into dataframe
    # X = pd.DataFrame([[u_p_total, uxp_reorder_ratio, u_total_orders,u_reordered_ratio,p_total,p_reordered_ratio]], 
    #                  columns = ["u_p_total", "uxp_reorder_ratio", "u_total_orders",'u_reordered_ratio',"p_total","p_reordered_ratio"])

    
    # # Get prediction
    # prediction = clf.predict(X)[0]
    
    # # Output prediction
    # st.text(f"This product will be {prediction}")
