# pip install -r /path/to/requirements.txt

import streamlit as st
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




# If button is pressed
if st.button("Submit"):

    # Load the model and tokenizer
    tokenizer = pickle.load(open('../model/tokenizer_avi.pkl', 'rb'))
    # load json and create model
    json_file = open('../model/nextword.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("../model/nextword_avi.h5")

    text= 'adventures with' 
    text = text.split(" ")
    text= text[-1]

    text = ''.join(text)
    predicted_words = Predict_Next_Words(loaded_model, tokenizer, text)

    st.write(predicted_words)


        # predicted_word='adventures,house,the,toy,dog'
        # st.text(f"This product will be {predicted_word}")

        # text2=st.text_input("Which predicted word would like to go forward with?")

        # if text2=='adventures':
        #     st.text(f"Movie Recommendation: Frankie Starlight, The quirky story of a young boy's adventures growing up with his stunningly beautiful mother and the two very different men who love her.")
        #     st.text(f"Movie Recommendation: दिल से, The clash between love and ideology is portrayed in this love story between a radio executive and a beautiful revolutionary.")
        #     st.text(f"Movie Recommendation: Zebrahead, Interracial love story set in Detroit.")

        # elif text2=='dog':
        #     st.text(f"Movie Recommendation: The Dog Problem, In the film, Solo is a down-on-his-luck writer who is encouraged by his psychiatrist to get a dog. Solo meets his love interest, who he assumes to be a dog owner when meeting her at a dog play park, but dog problems stand in their way.")
        #     st.text(f"Movie Recommendation: Good Boy!, An intergalactic dog pilot from Sirius (the dog star), visits Earth to verify the rumors that dogs have failed to take over the planet.")
        #     st.text(f"Movie Recommendation: Hachi: A Dog's Tale, A drama based on the true story of a college professor's bond with the abandoned dog he takes into his home.")















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
