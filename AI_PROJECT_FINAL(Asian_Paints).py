#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# In[4]:


data = pd.read_csv('ASIANPAINT.csv', date_parser = True)
data.tail()


# In[5]:


data_training = data[data['Date']<'2019-01-01'].copy()
data_test = data[data['Date']>='2019-01-01'].copy()


# In[6]:


data_training = data_training.drop(['Date','Prev Close','Last','Symbol','Series','VWAP','Turnover','Trades','Deliverable Volume','%Deliverble'], axis = 1)


# In[ ]:





# In[7]:


scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training)
data_training


# In[8]:


data_training[0:10]


# In[9]:


X_train = []
y_train = []


# In[10]:


for i in range(60, data_training.shape[0]):
    X_train.append(data_training[i-60:i])
    y_train.append(data_training[i, 0])


# In[11]:


X_train, y_train = np.array(X_train), np.array(y_train)


# In[12]:


X_train.shape


# In[13]:


import tensorflow as tf


# In[14]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


# In[ ]:





# In[15]:


regressior = Sequential()

regressior.add(LSTM(units = 60, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units = 120, activation = 'relu'))
regressior.add(Dropout(0.2))

regressior.add(Dense(units = 1))


# In[16]:


regressior.summary()


# In[17]:


regressior.compile(optimizer='adam', loss = 'mean_squared_error')


# In[18]:


regressior.fit(X_train, y_train, epochs=50, batch_size=32)


# In[19]:


data_test.head()


# In[20]:


data_test.tail(60)


# In[21]:


past_60_days = data_test.tail(60)


# In[22]:


df = past_60_days.append(data_test, ignore_index = True)
df = df.drop(['Date','Prev Close','Last','Symbol','Series','VWAP','Turnover','Trades','Deliverable Volume','%Deliverble'], axis = 1)
df.head()


# In[23]:


inputs = scaler.transform(df)
inputs


# In[24]:


X_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])


# In[25]:


X_test, y_test = np.array(X_test), np.array(y_test)
X_test.shape, y_test.shape


# In[27]:


y_pred = regressior.predict(X_test)


# In[28]:


scaler.scale_


# In[29]:


scale = 1/8.18605127e-04
scale


# In[30]:


y_pred = y_pred*scale
y_test = y_test*scale


# In[62]:


plt.figure(figsize=(14,5))
plt.plot(y_test, color = 'red', label = 'Real AsianPaints Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted AsianPaints Stock Price')
plt.title('AsianPaints Price Prediction')
plt.xlabel('Time')
plt.ylabel('AsianPaints Stock Price')
plt.legend()
plt.show()


# In[63]:


def graph(y_test,y_pred):

    plt.figure(figsize=(14,5))
    plt.plot(y_test, color = 'red', label = 'Real AsianPaints Stock Price')
    plt.plot(y_pred, color = 'blue', label = 'Predicted AsianPaints Stock Price')
    plt.title('AsianPaints Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('AsianPaints Stock Price')
    plt.legend()
    plt.show()


# In[34]:



from tkinter import * 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)


# In[60]:


def plot2():
    graph(y_test,y_pred)


# In[61]:


'''def plot():
  
    # the figure that will contain the plot
    fig = Figure(figsize = (5, 5),
                 dpi = 100)
  
    # list of squares
    y = [i**2 for i in range(101)]
  
    # adding the subplot
    plot1 = fig.add_subplot(111)
  
    # plotting the graph
    plot1.plot(y)
  
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,
                               master = window)  
    canvas.draw()
  
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()
  
    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,
                                   window)
    toolbar.update()
  
    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()'''
  
# the main Tkinter window
window = Tk()
  
# setting the title 
window.title('Plotting in Tkinter')
  
# dimensions of the main window
window.geometry("500x500")
  
# button that displays the plot
plot_button = Button(master = window, 
                     command = plot2,
                     height = 2, 
                     width = 10,
                     text = "Plot")
  
# place the button 
# in main window
plot_button.pack()
  
# run the gui
window.mainloop()


# In[55]:


# Import required libraries
from tkinter import *
from PIL import ImageTk, Image

# Create an instance of tkinter window
win = Tk()

# Define the geometry of the window
win.geometry("700x500")

frame = Frame(win, width=300, height=300)
frame.pack()
frame.place(anchor='center', relx=0.5, rely=0.5)

# Create an object of tkinter ImageTk
img = ImageTk.PhotoImage(Image.open("home.png"))

# Create a Label Widget to display the text or Image
label = Label(frame, image = img)
label.pack()

win.mainloop()


# In[57]:


from joblib import parallel,delayed
import joblib
joblib.dump(regressior, 'model.pkl')


# In[59]:


import streamlit as st


# In[ ]:




