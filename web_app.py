"""The main file to predict soccer world cup predictions

The code consists of 3 main parts:
1. Assigning each team data from the csv files to eact team name, and build a dataframe similar to the one the model is trained on.

2. Loading the model and the Preprocess pipeline to process the data including (one hot encoding for categorical variables and Making
PCA for the numerical values for dimensionality reduction

3.Making a web app front end frame work using Streamlit library to deploy the model and put it into production
"""

#Importing the libraries
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder


#loading in the model and the pipeline files to predict on the data for the first model
pickle_in = open('xgboost_hour_model.pkl', 'rb')
model = pickle.load(pickle_in)
classes = model.classes_

encoder1 = open('encoder.pkl', 'rb')
encoder = pickle.load(encoder1)


#create choose list for first model including the teams names from the trained data
district = [20,  2, 19,  1,  8,  9, 25,  6, 15,  3, 17, 10, 22, 16, 12, 18, 24, 11,  4,  5,  7, 14]

primary_type = ['BATTERY', 'THEFT', 'CRIMINAL DAMAGE', 'OTHER OFFENSE',
       'DECEPTIVE PRACTICE', 'ASSAULT', 'BURGLARY', 'MOTOR VEHICLE THEFT',
       'NARCOTICS', 'ROBBERY', 'PUBLIC PEACE VIOLATION',
       'CRIMINAL TRESPASS', 'PROSTITUTION', 'WEAPONS VIOLATION',
       'OFFENSE INVOLVING CHILDREN', 'CRIM SEXUAL ASSAULT', 'SEX OFFENSE',
       'INTERFERENCE WITH PUBLIC OFFICER', 'LIQUOR LAW VIOLATION',
       'CRIMINAL SEXUAL ASSAULT', 'KIDNAPPING', 'GAMBLING', 'ARSON']

zipcodes= [60653, 60612, 60643, 60619, 60628, 60623, 60645, 60625, 60638,
       60636, 60649, 60617, 60632, 60621, 60626, 60609, 60608, 60660,
       60629, 60613, 60611, 60641, 60640, 60624, 60620, 60637, 60647,
       60651, 60654, 60631, 60646, 60639, 60603, 60630, 60610, 60827,
       60644, 60618, 60634, 60601, 60615, 60614, 60633, 60652, 60302,
       60657, 60456, 60605, 60616, 60607, 60602, 60622, 60666, 60655,
       60402, 60707, 69063, 60606, 60656, 60659, 60661, 60642,  9020,
       60805, 60604]
month = [ 7,  3,  9,  5,  8,  1,  2, 10, 11,  6, 12,  4]
day_week = [5, 3, 2, 4, 7, 1, 6]

                
def welcome():
	return 'welcome all'


# this is the main function in which we define our webpage
def main():
	# giving the webpage a title
	
	# here we define some of the front end elements of the web page like
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;">Chicago Crime Hour Prediction App </h1>
	</div>
	"""
	
	# this line allows us to display a drop list to choose team 1 and team 2 
	st.markdown(html_temp, unsafe_allow_html = True)
	day = st.selectbox('Day of week', np.array(day_week))
	month1 = st.selectbox('Month', np.array(month))
	district1 = st.selectbox('District', np.array(district))
	zip_code =st.selectbox('Zip code', np.array(zipcodes))
	primary_type1 = st.selectbox('Primary type', np.array(primary_type))
	
	

        # the below line ensures that when the button called 'Predict' is clicked,
	# the prediction function defined above is called to make the prediction
	# and store it in the variable result
	results_df = pd.DataFrame()
	
	if st.button("Predict"):

            results_df = predict_match_result_goals(day,month1, zip_code, district1,primary_type1)
            st.markdown("Propability of crime at each hour")
            st.dataframe(results_df)	    

            fig, ax = plt.subplots()
            ax.bar(results_df.columns,results_df.iloc[0])
            ax.set_xticks(results_df.columns)
            st.markdown("Propability of crime at each hour Distribution")
            st.pyplot(fig)



#Predict function for the final match result prediction
def predict_match_result_goals(day_of_week3,month3, zipcode3, district3,p_type):

    p_type1 = encoder.transform(pd.DataFrame([p_type]).T)
    input_data = pd.DataFrame([p_type1[0],district3,zipcode3,month3,day_of_week3]).T
    preds = model.predict_proba(input_data)
    df5 = pd.DataFrame(columns=model.classes_,data=np.round(preds,3))

    return df5
    
	
if __name__=='__main__':
	main()

