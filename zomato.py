import streamlit as st
import joblib
import pandas as pd
import numpy as np


select_page = st.sidebar.radio('Select page', ['Introduction', 'Model Classification'])

if select_page == 'Introduction':
    def main():
        st.title('Zomato Bangalore Restaurants')
        st.write('### Introduction to the Data:')
        st.write('''In the vibrant culinary landscape of Bangalore, Zomato serves as a crucial platform for food lovers...''')
        
        st.header('Dataset Feature Overview')
        st.write('''
            *address*: contains the address of the restaurant in Bengaluru.
            *name*: contains the name of the restaurant.
            *online_order*: whether online ordering is available in the restaurant or not.
            *book_table*: table booking option available or not.
            *rate*: contains the overall rating of the restaurant out of 5.
            *votes*: contains total number of ratings for the restaurant as of the above mentioned date.
            *phone*: contains the phone number of the restaurant.
            *location*: contains the neighborhood in which the restaurant is located.
            *rest_type*: restaurant type.
            *dish_liked*: dishes people liked in the restaurant.
            *cuisines*: food styles, separated by commas.
            *approx_cost(for two people)*: contains the approximate cost for a meal for two people.
            *reviews_list*: list of tuples containing reviews for the restaurant.
            *menu_item*: contains list of menus available in the restaurant.
            *listed_in(type)*: type of meal.
            *listed_in(city)*: contains the neighborhood in which the restaurant is listed.
        ''')

    if __name__ == '__main__':
        main()



if select_page == 'Model Classification':
    def main():
        st.title('Model Classification')
        pipeline = joblib.load('RF_pipeline.pkl')

        def Prediction(address, name, online_order, book_table, location, rest_type, dish_liked, cuisines, menu_item, dining_type, location_city, approx_cost):
            df = pd.DataFrame(columns=['address', 'name', 'online_order', 'book_table', 'location', 'rest_type', 'dish_liked', 'cuisines', 'menu_item', 'dining_type', 'location_city'])
            df.at[0, 'address'] = address
            df.at[0, 'name'] = name
            df.at[0, 'online_order'] = online_order
            df.at[0, 'book_table'] = book_table
            df.at[0, 'location'] = location
            df.at[0, 'rest_type'] = rest_type
            df.at[0, 'dish_liked'] = dish_liked
            df.at[0, 'cuisines'] = cuisines
            df.at[0, 'menu_item'] = menu_item
            df.at[0, 'dining_type'] = dining_type
            df.at[0, 'location_city'] = location_city
            try:
                approx_cost = float(approx_cost)  # Convert approx_cost to float
            except ValueError:
                st.error('Please enter a valid number for Approx Cost.')
                return None  # Stop execution if invalid input

            df.at[0, 'approx_cost(for two people)'] = approx_cost

            result = pipeline.predict(df)[0]
            return result

       
        address = st.text_input('Please write your address')
        name = st.text_input('Please write your name')
        online_order = st.selectbox('Is online ordering available?', ['Yes', 'No'])
        book_table = st.selectbox('Is table booking option available?', ['Yes', 'No'])
        location = st.text_input('Please write your location')
        rest_type = st.text_input('Please write your restaurant type')
        dish_liked = st.text_input('Please write the dish liked')
        cuisines = st.text_input('Please write your cuisines')
        menu_item = st.text_input('Please write your menu items')
        dining_type = st.text_input('Please write your dining type') 
        location_city = st.text_input('Please write your location city')
        approx_cost = st.text_input('Please enter the approximate cost for two people')

        if st.button('Predict'):
            result = Prediction(address, name, online_order, book_table, location, rest_type, dish_liked, cuisines, menu_item, dining_type, location_city, approx_cost)
            if result is not None:
                st.write('### Prediction Result:')
                st.write(f'The predicted result is: {round(np.exp(result), 2)}')

    if __name__ == '__main__':
        main()
