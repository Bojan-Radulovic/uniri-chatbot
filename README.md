# uniri-chatbot
A chatbot that answers questions about the University of Rijeka (mainly about FFRI and FIDIT)
## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Use](#use)
## General info
This project was created by me during my professional practice at Ericsson Nikola Tesla. It involved developing a Flask web app with a chat-like user interface that answers questions about the University of Rijeka and its's faculties. In the app, users can ask questions about the University of Rijeka, primarily focusing on FFRI and FIDIT, and receive AI-generated answers. The app also includes speech-to-text capabilities, allowing users to ask questions using their voice instead of typing. Additionally, it features text-to-speech capabilities to read the answers aloud to users. Users can also choose whether they want the chatbot to provide sources for the answers. The app currently supports only Croatian and incorporates information from the university and faculties' websites, freshman guides, pamphlets, etc. However, the data can be easily modified to create a chatbot for virtually any purpose.
## Technologies
* Python 3.9
* Flask
* LangChain
* OpenAI API
* ...
## Use
In the openaiBot.py file, replace "insert your OpenAI API key here" with your OpenAI API key. Extract the vectorstore.rar file in your root folder and, from a terminal in the root folder, start the server with python app.py.
