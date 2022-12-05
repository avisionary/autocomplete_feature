# pip install -r /path/to/requirements.txt

import streamlit as st
import pandas as pd
import joblib 
import plotly.express as px

#from tensorflow.keras.models import load_model
import numpy as np
import pickle



# Title
st.header("Get Movie Recommendations from an Auto Complete Thoughts Prediction App")



text=st.text_input("What's on your mind?")

# df = px.data.tips()
# fig = px.box(df, y="total_bill")
# fig.show()


# # Input bar 1
# u_p_total = st.number_input("Enter the number of times a user has bought the product")

# # Input bar 2
# uxp_reorder_ratio = st.number_input("Enter the estimated probability of how frequently a user bought the product")

# # Input bar 3
# u_total_orders= st.number_input("Enter the number of orders placed by a user")

# # Input bar 4
# u_reordered_ratio = st.number_input("Enter how frequently has a user reordered products")

# # Input bar 5
# p_total=st.number_input("Enter the number of times a product has been purchased")

# # Input bar 6
# p_reordered_ratio=st.number_input("Enter the estimated probability of porduct being reordered")

# What does the predicted value mean?
#st.text('1 means Reordered, 0 means Not reordered')

# If button is pressed
if st.button("Submit"):

#     model = load_model('nextword1.h5')
#     tokenizer = pickle.load(open('tokenizer1.pkl', 'rb'))
    
#     for i in range(3):
#         sequence = tokenizer.texts_to_sequences([text])[0]
#         sequence = np.array(sequence)
        
#         preds = model.predict(sequence)
#         pp=(-preds[0]).argsort()[:10]
#         #pp=pp[0:5]
#         #preds=np.argmax(preds,axis=1)
# #         print(preds)
#         predicted_word = []
        
#         for key, value in tokenizer.word_index.items():
#                for word in pp:   
#                        if value == word:
#                         predicted_word.append(key)
#                         break

        predicted_word='adventures,house,the,toy,dog'
        st.text(f"This product will be {predicted_word}")

        text2=st.text_input("Which predicted word would like to go forward with?")

        if text2=='adventures':
            st.text(f"Movie Recommendation: Frankie Starlight, The quirky story of a young boy's adventures growing up with his stunningly beautiful mother and the two very different men who love her.")
            st.text(f"Movie Recommendation: दिल से, The clash between love and ideology is portrayed in this love story between a radio executive and a beautiful revolutionary.")
            st.text(f"Movie Recommendation: Zebrahead, Interracial love story set in Detroit.")

        elif text2=='dog':
            st.text(f"Movie Recommendation: The Dog Problem, In the film, Solo is a down-on-his-luck writer who is encouraged by his psychiatrist to get a dog. Solo meets his love interest, who he assumes to be a dog owner when meeting her at a dog play park, but dog problems stand in their way.")
            st.text(f"Movie Recommendation: Good Boy!, An intergalactic dog pilot from Sirius (the dog star), visits Earth to verify the rumors that dogs have failed to take over the planet.")
            st.text(f"Movie Recommendation: Hachi: A Dog's Tale, A drama based on the true story of a college professor's bond with the abandoned dog he takes into his home.")















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
