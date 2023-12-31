## Sentiment Analysis

This repository contains code for sentiment analysis using Natural Language Processing (NLP) techniques. The goal of sentiment analysis is to determine the sentiment or emotional tone of a piece of text, whether it is positive, negative, or neutral.

#### Database

The sentiment analysis model is trained on a labeled dataset containing text samples along with their corresponding sentiment labels. The dataset is preprocessed and tokenized to convert the text into numerical representations suitable for input to the model. 

The data we were used still has two main dimensions: tweets and labels. The tweets represent pure text collected from Twitter about ChatGPT, covering various topics and opinions shared by users. The labels associated with each tweet indicate the sentiment that the authors want to transmit in their messages, such as positive, negative, neutral.

To access and explore the dataset, please follow this link: [ChatGPT-Tweets](https://www.kaggle.com/datasets/charunisa/chatgpt-sentiment-analysis). 
Please, Download and copie the dataset to databse dir.


#### Model Architecture

The sentiment analysis model uses a deep learning architecture with an Embedding layer followed by one or more Recurrent Neural Network (RNN) layers, such as Long Short-Term Memory (LSTM). The model employs an attention mechanism to focus on important parts of the input text.

#### Training and Evaluation

The model is trained using a training set and evaluated on a separate validation set to assess its performance. We monitor various metrics such as accuracy, precision, recall, and F1-score to gauge the model's effectiveness in classifying sentiments correctly.

#### Dependencies

The libraries presented in requeriments.txt are required to run the sentiment analysis code. You can install these dependencies using pip with the following command: 
* pip install -r requirements.txt or run make install command
I recommend you create a virtual env before install dependencies.
