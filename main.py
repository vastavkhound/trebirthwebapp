import streamlit as st
from firebase import  firebase
from scipy import signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import pandas as pd
from scipy.fft import fftshift,fft, fftfreq, rfft, rfftfreq
import os
import glob
import json
from PIL import Image
import csv
import pymysql
import boto3
import json


# s3_client = boto3.client('s3')
# response=s3_client.get_object(Bucket='trebirth1',Key='FB.json')
# result = response['Body'].read()
# print("Result is",result)
#retrieve json file from firebase
firebase = firebase.FirebaseApplication('https://esp32-d544d-default-rtdb.firebaseio.com/',None)
result = firebase.get("test","sensor")
result1 = firebase.get("test","scan_number")
Np_result = np.array(result)
df = pd.DataFrame(Np_result)
df.to_csv( "scan.csv")



jtopy=json.dumps(result)       #json.dumps take a dictionary as input and returns a string as output.
dict_json=json.loads(jtopy)    # json.loads take a string as input and returns a dictionary as output.
# print(dict_json)

# Digital Filter starts from here
def Data_Preprocess(x):
 sig = [np.array(x)]
 # print("Sig is ",sig)
 return sig

def Apply_Filter(sig):
    sos = signal.butter(1, [0.1,10], 'band', fs=100, output='sos')
    filtered = signal.sosfilt(sos, sig)
    # print ("Filtered data is ",filtered)
    return filtered.squeeze()


def Plot_Graph(filtered):
   t = np.linspace(0, 30,3000, False)
   t = t[:filtered.size]
   fig = px.line(x=t, y=filtered, labels={'x':'Time', 'y':'Amplitude'},title='Time Series', width = 1000, height = 600)
   fig.update_traces(line_color='#2AA6FB', line_width=1.5)
   st.plotly_chart(fig, use_container_width=False, sharing="streamlit")

   
   
   
def Calculate_FFT(sig_data):
   N = 1500
   yf = rfft(sig_data[:1500])
   xf = rfftfreq(N, 0.01)
  # fig, ax = plt.subplots()
   #ax.plot(xf,np.abs(yf))
   #ax.grid()
   #plt.axis([0, 10, 0, 100000])
   #plt.suptitle('Fourier Transform of the Signal')
   #plt.xlabel("Frequency(Hz)")
   #plt.ylabel("Amplitude")
   #st.pyplot(fig)
   yf = yf[:40000]
   fig = px.line(x=xf, y=np.abs(yf), labels={'x':'Frequency(Hz)', 'y':'Amplitude'},title='Fourier Transform', width = 1000, height = 600, markers=True)
   fig.update_traces(line_width=1.5)
   st.plotly_chart(fig, use_container_width=False, sharing="streamlit")
   #plt.plot(xf, yf)
   #plt.show()


   #plt.plot(t.squeeze(),filtered.squeeze())
  # plt.suptitle('Filtered Scan Data')
   #plt.axis([0, 15, 0, 400])
   #plt.savefig("output.jpg")

Data = Data_Preprocess(dict_json)
# print("Data is ",Data)
Filtered_data = Apply_Filter(Data)
# print(Filtered_data)
# plt.show()
plt.savefig("output.jpg")



#Streamlit GUI starts from here
a=st.sidebar.radio('Navigation',['Farm Information','Farmer Data'])
# df = pd.read_csv("Trebirth.csv")

if a == "Farm Information":
 st.header("Welcome to Trebirth Tech Development")
#  form = st.form(key='my_form',clear_on_submit=True)
#  F_name= form.text_input(label='Enter Farmer Name')
#  F_health= form.text_input(label='Enter Farm Health')
#  Number= form.number_input(label='Enter No. of trees scanned')
#  Remark = form.text_area(label='Remark')
#  submit_button = form.form_submit_button(label='Submit')


#  st.sidebar.markdown(
#     f"""
#      * Farmer name :        {F_name}
#      * Farm health :        {F_health}
#      * No of trees scanned: {Number}
#      * Remark      :        {Remark}
#  """
#   )
 st.subheader(f'Scan number is: {result1}')
 #st.write("Scan number is ", result1)
 Plot_Graph(Filtered_data)
 print(Filtered_data) 
 #st.line_chart(Filtered_data)
 Calculate_FFT(Np_result)
 #st.line_chart(Filtered_data, width=1000, height=0, use_container_width=False)
 st.write(df)
 


#  if submit_button:
#      st.write(F_name,F_health,Number,Remark)
#  new_data = {"Farmer_Name": F_name,"Farm_Health": F_health,"Trees_Scanned": int(Number),"Remark": Remark}
#  #st.write(new_data)
#  df = df._append(new_data,ignore_index=True)
#  df.to_csv("Trebirth.csv",index=False)
# st.dataframe(df)


@st.cache
def convert_df(df):
 return df.to_csv().encode('utf-8')
csv = convert_df(df)

st.download_button(
     "Press to Download",
     csv,
     "file.csv",
     "text/csv",
     key='download-csv'
 )





 # col1, col2= st.columns(2)
 #
 # with col1:
 #     st.header("Filtered Data")
 #     st.line_chart(Filtered_data)

 # with col2:
 #     st.header(" Accelerometer")
 #     st.line_chart(Filtered_data)






