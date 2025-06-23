# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 18:59:03 2025

@author: ankit
"""

import numpy as np
import pickle
import streamlit as st


def predictive_system(input_data):
    
    import numpy as np
    import pandas as pd
    import difflib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import pickle

    # Load the model and data
    loaded_model = pickle.load(open('C:/Users/ankit/OneDrive/Documents/Machine-Learning/movie_recommendation/trained_model.sav', 'rb'))

    movie_data = pd.read_csv('C:/Users/ankit/OneDrive/Documents/Machine-Learning/movie_recommendation/movies.csv')  # Adjust path if needed

    # Preprocessing
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movie_data[feature] = movie_data[feature].fillna('')

    combined_features = movie_data['genres'] + ' ' + movie_data['keywords'] + ' ' + \
                        movie_data['tagline'] + ' ' + movie_data['cast'] + ' ' + movie_data['director']

    # Transform features
    feature_vector = loaded_model.transform(combined_features)

    # Compute similarity
    similarity = cosine_similarity(feature_vector)

    # Get user input
    movie_name = input_data

    list_of_titles = movie_data['title'].tolist()
    close_match = difflib.get_close_matches(movie_name, list_of_titles)

    if close_match:
        close = close_match[0]
        index = movie_data[movie_data.title == close].index[0]

        similarity_score = list(enumerate(similarity[index]))
        sorted_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        print("Movies suggested for you:\n")
        
        
        recommendations = [] 
        i = 1

        for movie in sorted_movies:
            idx = movie[0]
            title = movie_data.iloc[idx]['title']
            if i <= 30:
                recommendations.append(f"{i}. {title}")
                i += 1

        # Convert list to NumPy array
        recommendation_array = np.array(recommendations)
        
        return recommendation_array
        
        

def main():
    
    st.title('Movie recommendation system')
    
    input_data = st.text_input('Enter your favourite movie name: ')
    
    result = []
    
    if st.button('Movies Recommended are '):
        result = predictive_system(input_data)
        
    for movie in result:
            st.success(movie)
    
    

if __name__=='__main__':
    main()
    
    
    