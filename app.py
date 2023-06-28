import base64
from flask import Flask, render_template, request
from queue import Queue
from threading import Thread
from openaiBot import dataFromPickle, docsFromPdfs, docsFromUrls, createFAISSVectorstore, createPickle
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from markupsafe import Markup
import os
from gtts import gTTS
import uuid

app = Flask(__name__)
sessions = {}
if os.path.isfile("vectorstore.pkl"):
    vectorstore = dataFromPickle("vectorstore.pkl")
else:
    docs = docsFromUrls('links.txt') + docsFromPdfs('./pdfs')
    vectorstore = createFAISSVectorstore(docs)
    createPickle(vectorstore, "vectorstore.pkl")

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), return_source_documents=True)


def response_listener(session_key):
    chat_history = []
    while True:
        user_input = sessions[session_key]["input_queue"].get()
        result = qa({"question": user_input, "chat_history": chat_history})
        chat_history.append((user_input, result["answer"]))
        sessions[session_key]["response_queue"].put(result)


@app.route('/')
def home():
    session_key = str(uuid.uuid4())
    sessions[session_key] = {"chat": [], "input_queue": Queue(), "response_queue": Queue()}

    response_thread = Thread(target=response_listener, args=(session_key,))
    response_thread.start()

    initial_greeting = "Dobar dan. Kako Vam mogu pomoći?"
    sessions[session_key]["chat"].append(('bot', initial_greeting))
    output_path = os.path.join(app.root_path, 'output.mp3')
    tts = gTTS(text=initial_greeting, lang='hr')
    tts.save(output_path)

    with open(output_path, 'rb') as f:
        audio_content = base64.b64encode(f.read()).decode('utf-8')

    return render_template('index.html',
                           chat=sessions[session_key]["chat"],
                           show_sources=False,
                           audio_content=audio_content,
                           session_key=session_key)


@app.route('/', methods=['POST'])
def generate_response():
    user_input = request.form['user_input']
    show_sources = request.form.get('show_sources')
    session_key = request.form['session_key']

    sessions[session_key]["chat"].append(('user', user_input))

    sessions[session_key]["input_queue"].put(user_input)

    response = sessions[session_key]["response_queue"].get()

    sessions[session_key]["chat"].append(('bot', response["answer"][1:]))
    if show_sources:
        sources_string = "Izvori: "
        for n, sources in enumerate(response["source_documents"]):
            sources_string += Markup(f"<br>{n+1}) <span class='source-text'>" + sources.metadata["source"] + "</span>")
        sessions[session_key]["chat"].append(('bot', sources_string))

    output_path = os.path.join(app.root_path, 'output.mp3')
    tts = gTTS(text=response["answer"][1:], lang='hr')
    tts.save(output_path)

    with open(output_path, 'rb') as f:
        audio_content = base64.b64encode(f.read()).decode('utf-8')

    return render_template('index.html',
                           chat=sessions[session_key]["chat"],
                           show_sources=show_sources,
                           audio_content=audio_content,
                           session_key=session_key)


if __name__ == '__main__':
    app.run()