import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from GoogleNews import GoogleNews
from textblob import TextBlob
model = load_model('Stock Predictions Model.keras')

st.header('Stock Market Predictor')

option = st.selectbox(
    'Choose an option:',
    ('Stock Predictions', 'Sentimental Analysis'))

if option == 'Stock Predictions':
    stock = st.text_input('Enter Stock Symbol', 'AAPL')

    
    start_date = st.date_input('Start Date', pd.Timestamp('2024-01-14'))
    end_date = st.date_input('End Date', pd.Timestamp('2025-01-14'))

    data = yf.download(stock, start_date, end_date)

    st.subheader('Stock Data')
    st.write(data)

    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    scaler = MinMaxScaler(feature_range=(0,1))
    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    st.subheader('Price vs MA50')
    ma_50_days = data.Close.rolling(50).mean()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data.index, y=ma_50_days, mode='lines', name='MA50', line=dict(color='red')))
    fig1.add_trace(go.Scatter(x=data.index, y=data.Close, mode='lines', name='Close', line=dict(color='green')))
    fig1.update_layout(title='Price vs MA50', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig1)

    ma_100_days = data.Close.rolling(100).mean()
    
    # Plotting MA50 vs MA100
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=ma_50_days, mode='lines', name='MA50', line=dict(color='red')))
    fig2.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100', line=dict(color='blue')))
    fig2.update_layout(title='MA50 vs MA100', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig2)

    ma_200_days = data.Close.rolling(200).mean()
    # Plotting MA100 vs MA200
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100', line=dict(color='blue')))
    fig3.add_trace(go.Scatter(x=data.index, y=ma_200_days, mode='lines', name='MA200', line=dict(color='orange')))
    fig3.update_layout(title='MA100 vs MA200', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig3)

        


    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])

    x,y = np.array(x), np.array(y)

    predict = model.predict(x)

    scale = 1/scaler.scale_

    predict = predict * scale
    y = y * scale

    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=np.arange(len(predict)), y=predict[:,0], mode='lines', name='Original Price', line=dict(color='red')))
    fig4.add_trace(go.Scatter(x=np.arange(len(y)), y=y, mode='lines', name='Predicted Price', line=dict(color='green')))
    fig4.update_layout(title='Original Price vs Predicted Price', xaxis_title='Time', yaxis_title='Price')
    st.plotly_chart(fig4)

elif option == 'Sentimental Analysis':
    st.title('Sentimental Analysis')
    stock = st.text_input('Enter Your Stock', 'AAPL')


    start_date = st.date_input('Start Date', pd.to_datetime('2023-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2024-04-30'))

    googlenews = GoogleNews()
    googlenews.search(stock)
    result = googlenews.result()
    news_text = ''
    for res in result:
        news_text += res['title'] + '\n'
    st.subheader('News')
    st.write(news_text)
    blob = TextBlob(news_text)
    sentiment_score = blob.sentiment.polarity
    st.subheader('Sentiment Analysis Result')
    if sentiment_score > 0:
        st.write('Overall sentiment: Positive')
        st.write('Verdict: Investable')
    elif sentiment_score < 0:
        st.write('Overall sentiment: Negative')
        st.write('Verdict: Not Investable')
    else:
        st.write('Overall sentiment: Neutral')
        st.write('Verdict: Neutral')

   
    data = yf.download(stock, start_date, end_date)
    fig_stock = go.Figure()
    fig_stock.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Stock Price'))
    fig_stock.update_layout(title='Stock Price Movement', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_stock)
