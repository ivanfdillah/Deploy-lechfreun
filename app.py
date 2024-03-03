import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
from streamlit_option_menu import option_menu
import pickle
import tensorflow as tf
from keras.models import load_model



selected1 = option_menu(menu_title = None, options = ["Home","Model", "Our Team"],  orientation = "horizontal")
   
if selected1 == "Home":
   st.image('image/a.jpg')
   st.title('Automatic Systematic Literature Review')
   st.write("""Selamat datang di Automatic Systematic Literature Review. Aplikasi ini menggunakan
         teknologi deep learning untuk mengidentifikasi berbagai kata dari abstrack dan title. 
         Dengan menggunakan algoritma klasifikasi canggih, aplikasi ini dapat 
         membantu dokter dan peneliti dalam mengklasifikasikan title dan abstract .""")

   st.header("Cara Penggunaan")
   st.text("1. Masukkan keyword untuk melakukan prediksi")
   st.text("2. Click tombol 'Prediksi'")
   st.text("3. Tunggu hasil prediksi dan lihat hasil predksi")

   st.header("Metode dan Tujuan dari website ini", divider="gray")
   col1, col2 = st.columns(2)
   with col1:
      st.subheader("Teknologi Ai")
      st.write("""Aplikasi ini menggunakan teknologi berbasi AI
             dengan menggunakan metode deep learning dan
            feature representation tfidf""")
   
   with col2:
      st.subheader("Screening Artikel")
      st.write("""Aplikasi ini dapat membantu para dokter dan 
            peneliti untuk membantu screening artikel dengan efisien""")   
   
elif selected1 == "Model":
      
   st.header('Aplikasi Klasifikasi Keyword')  
   model = load_model('model_cnn_k-fold.h5')
   vocab = pickle.load(open('k-beast_feature.pickle', 'rb'))

   tf_idf_vec = TfidfVectorizer(vocabulary=set(vocab))
 

   options = ("Title", "Abstract")
   selected_option = st.selectbox("Pilih Opsi:", options, index=None)

   if selected_option == "Title":
      user_input = st.text_input('Masukkan Title :')
      tfidf = tf_idf_vec.fit_transform([user_input])
      dense_array = tfidf.toarray()
      prediksi = model.predict(dense_array)
      threshold_value = 0.58
      binary_prediction = (prediksi > threshold_value).astype('int')
     # prediction_label = 'include' if binary_prediction[0] == 1 else 'exclude'
      if st.button("prediksi"):
        
         if binary_prediction[0] == 1:
            prediction = 'include'
         else:
            prediction = 'exclude'

         st.write("Hasil prediksi: adalah", prediction)

   elif selected_option == "Abstract":
      user_input1 = st.text_input('Masukkan Abstract :')
      tfidf = tf_idf_vec.fit_transform([user_input1])
      dense_array = tfidf.toarray()
      prediksi = model.predict(dense_array)
      threshold_value = 0.58
      binary_prediction = (prediksi > threshold_value).astype('int')
     # prediction_label = 'include' if binary_prediction[0] == 1 else 'exclude'
      if st.button("prediksi"):
        
         if binary_prediction[0] == 1:
            prediction = 'include'
         else:
            prediction = 'exclude'

         st.write("Hasil prediksi: adalah", prediction)


elif selected1 == "Our Team":
   st.header('Our Teams')
   st.write("""Dalam proses pembuatan website aplikasi klasifikasi keyword 
         disini kami terdiri dari 3 orang""")

   col1, col2, col3 = st.columns(3)
   with col1:
      original_title = '<p font-size: 40px;"></p>'
      st.markdown(original_title, unsafe_allow_html=True)
      st.image('image/Picture1.png', caption='Dikco Agung Prasetyo')

   with col2:
      original_title = '<p font-size: 40px; "></p>'
      st.markdown(original_title, unsafe_allow_html=True)
      st.image('image/Picture2.png', caption='Muhamad Ivan Fadhillah')

   with col3:
      original_title = '<p font-size: 40px;"></p>'
      st.markdown(original_title, unsafe_allow_html=True)
      st.image('image/Picture3.png', caption='Muhammad Soleh Apriadi')

