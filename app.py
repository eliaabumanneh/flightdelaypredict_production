import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.utils import custom_object_scope
np.random.seed(0)
from datetime import datetime
from pathlib import Path
from time import sleep
from flask import Flask, render_template, request
from pathlib import Path

#Initialises the flask app
app = Flask(__name__)

#Helper functions
def rmse(y_true, y_pred): #defining the Root Mean Squared Error function
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def yeartodate_scaled():
    day_of_year = datetime.now().timetuple().tm_yday
    return day_of_year / 365

def data_setup(): #CSV and H5 file imports and
    #creating the base_path for future relative paths
    base_path = Path(__file__).parent

    airports_df = pd.read_csv((base_path / "airports.csv").resolve())
    airlines_df = pd.read_csv((base_path / "airlines.csv").resolve())
    ontime_10423 = pd.read_csv((base_path / "ontime_10423.csv").resolve())


    with custom_object_scope({'rmse': rmse}): #adding our rmse helper function
        global modely1, modely2
        modely1 = load_model((base_path / "modely1.h5").resolve())
        modely2 = load_model((base_path / "modely2.h5").resolve())

    #Setting up the input matrix
    X_data = ontime_10423.iloc[:,:-64]
    X_data.drop(['ORIGIN_AIRPORT_ID','DEP_DELAY','CANCELLED'], axis=1, inplace=True)
    collist = X_data.columns.tolist()

    global airline_list, airport_list

    #creating a list of airlines for users to pick from
    airline_list = []
    for col in collist:
        if col.startswith('OP_UNIQUE_CARRIER_'):
            airline_list.append(col.replace('OP_UNIQUE_CARRIER_', ''))

    #creating a dictionary to map OP_UNIQUE_CARRIER to CARRIER_NAME
    carrier_dict = airlines_df.set_index('OP_UNIQUE_CARRIER')['CARRIER_NAME'].to_dict()
    
    #using the map function to replace the values in airline_list
    airline_list = [carrier_dict.get(airline, airline) for airline in airline_list]

    #creating a list of destination airports for users to pick from
    airport_list = []
    for col in collist:
        if col.startswith('DEST_AIRPORT_ID_'):
            airport_list.append(col.replace('DEST_AIRPORT_ID_', ''))
    airport_list = pd.Series(airport_list).astype('int64').tolist()

    #creating a dictionary to map AIRPORT_ID to DISPLAY_AIRPORT_NAME
    airport_dict = airports_df.set_index('AIRPORT_ID')['DISPLAY_AIRPORT_NAME'].to_dict()

    #using the map function to replace the values in airport_list
    airport_list = [airport_dict.get(airport, airport) for airport in airport_list]

    return airport_list, airline_list, collist

#Sser input prediction function
def user_pred(numpy_array_input):  #input is shape (43,), all OHE except the last 

    #make delay prediction with the model
     #Unit testing the input
    raw_delay_prediction = modely1.predict(numpy_array_input)

    transformed_delay_prediction = np.exp(raw_delay_prediction) -30

    #make cancellation prediction with the model
    cancellation_prediction = modely2.predict(numpy_array_input)

    return transformed_delay_prediction, cancellation_prediction

#Main Prediction Function
def run_pred(input_dest, input_airline):

    global new_list
    new_list = [0]*len(collist) #Resetting input
    
    new_list[input_airline] = 1          #Executes the addition of airline to input
    new_list[input_dest+17] = 1          #Executes the addition of airport to input
    new_list[-1] = round(yeartodate_scaled(),3)  #Executes the addition of scaled YTD

    X_input = np.array(new_list).reshape(-1,43)

    #test prediction
    prediction = user_pred(X_input)
    delay_pred_array = (prediction[0])
    delay_pred = round(float(delay_pred_array[0]),2)

    #output to be sent to user
    cancellation_pred_array = (prediction[1])*100
    cancellation_pred = abs(round(float(cancellation_pred_array[0]),2))

    return delay_pred,cancellation_pred

loaded = 0 #Initial variable - 0 means data_setup() has not occurred yet

#Flask components
@app.route('/') #routes to html page at ('/')
def index():
    global airport_list, airline_list, collist, loaded, airport_default_index, airline_default_index
    if loaded ==0: #if data not loaded (0), loading occurs. This prevents repeat data_setup every time the ('/') page is visited.
        airport_list, airline_list, collist = data_setup()
        airport_default_index = 0
        airline_default_index = 0
        loaded = 1
    
    return render_template('webpage.html', airport_list=airport_list, airline_list = airline_list, airport_default_index=airport_default_index, airline_default_index=airline_default_index )

@app.route('/predict', methods=['GET','POST'], )
def predict(): # Make prediction based on selected values
    #Feature user input    
    int_features = [int(x) for x in request.form.values()]
    airport_index = int_features[0] 
    airline_index = int_features[1]

    #Setting the dropdown boxes to freeze the choice
    global airport_default_index, airline_default_index
    airport_default_index = airport_index
    airline_default_index = airline_index

    #Prediction Model running
    delay, cancellation = run_pred(airport_index, airline_index)

    return render_template('webpage.html',user_prompt='For a flight to {}'.format(airport_list[airport_index]) + ' on {}'.format(airline_list[airline_index]), prediction_text='Expected flight delay time is {}'.format(delay) + ' minutes', cancellation_text='The likelihood of cancellation is {}'.format(cancellation) + '%', airport_list=airport_list, airline_list=airline_list)

if __name__ == '__main__': 
    port = int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0', port=port)
    