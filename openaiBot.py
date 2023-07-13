from langchain.document_loaders import UnstructuredURLLoader, DirectoryLoader
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores.faiss import FAISS
import pickle
import os

openai_api_key = "insert your OpenAI API key here"
os.environ["OPENAI_API_KEY"] = openai_api_key

# namjestanje environmenta za tesseract i poppler
path = os.environ.get('PATH', '')
poppler_path = os.path.join(os.getcwd(), "poppler-23.07.0", "Library", "bin")
tesseract_path = os.path.join(os.getcwd(), "Tesseract-OCR")
tessdata_prefix = os.path.join(tesseract_path, "tessdata")
new_path = f'{path};{poppler_path};{tesseract_path}'
os.environ['PATH'] = new_path
os.environ['TESSDATA_PREFIX'] = tesseract_path

# korisno za izradu valstitih loadera,
# u ovom projektu ipak nije koristeno
class CustomLoader(BaseLoader):
    """Load text files."""
    def __init__(self, doc_list: str):
        """Initialize with file path."""
        self.doc_list = doc_list

    def load(self):
        """Load from file path."""
        return self.doc_list


def dataFromPickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# vjerojatno je potrebno ciscenje "smeca" iz izvora
# (uklanjanje irelevantnih i pokvarenih linkova i pdfova,
# uklanjnje headera itd.)
def docsFromUrls(path):
    def getLinks():
        with open(path, 'r') as file:
            contents = file.read()
            contents = contents.split("\n")
            contents = [s for s in contents if not s.startswith("#")]
            return contents

    urls = getLinks()
    loader = UnstructuredURLLoader(urls=urls)
    return loader.load()


def docsFromPdfs(path):
    loader = DirectoryLoader(path)
    return loader.load()


def createPickle(data, path):
    if os.path.isfile(path):
        os.remove(path)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


# nije potrebno jer FAISS ima metodu .add_documents()
def appendPickle(new_data, path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    data += new_data
    createPickle(data, path)


# vjerojatno je potrebno bolje podesiti splitter (npr. isprobati overlap=200)
def createFAISSVectorstore(docs, text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0), embeddings=OpenAIEmbeddings()):
    docs = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


# ova funkcija se koristi samo kada se bot zeli pokrenuti u terminalu
# (umjesto python app.py odkomentirati startbot() na dnu ove datoteke
# i pokrenuti python openaiBot())
def startBot():
    if os.path.isfile("vectorstore.pkl"):
        vectorstore = dataFromPickle("vectorstore.pkl")
    else:
        docs = docsFromUrls('links.txt') + docsFromPdfs('./pdfs')
        vectorstore = createFAISSVectorstore(docs)
        createPickle(vectorstore, "vectorstore.pkl")

    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), return_source_documents=True)
    chat_history = []
    print()
    print("Bot: Dobar dan, postavite mi pitanje.")
    while 1:
        print()
        query = input("Ja: ")
        print()
        if query.upper() == "EXIT":
            break
        result = qa({"question": query, "chat_history": chat_history})
        chat_history.append((query, result["answer"]))
        print("Bot:", result["answer"][1:])
        izvori = []
        for izvor in result["source_documents"]:
            izvori.append(izvor.metadata["source"])
        print("Izvori:", izvori)
        print("Bot: Zanima li Vas još nešto?")


# startBot()
