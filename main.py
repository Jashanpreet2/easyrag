# Connects with the LLM
from langchain_community.llms import Ollama
# Gets string output
from langchain_core.output_parsers import StrOutputParser
# Adds prompts
from langchain_core.prompts import ChatPromptTemplate
# Creates the document based chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Creates the retrieval chain
from langchain.chains.retrieval import create_retrieval_chain

# Loads website data
from langchain_community.document_loaders import WebBaseLoader
# Creates embeddings
from langchain_community.embeddings import OllamaEmbeddings

# The documents are indexed into this vector store
from langchain_community.vectorstores import FAISS
# Splits the text into separate paragraphs, then sentences, then words, and then letters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Can act as a document for the document chain
from langchain_core.documents import Document

# The parts of the chain
llm = Ollama(model="llama3")
output_parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template("""Answer the following question based on the provided context only:

<context>
{context}
</context>

Question: {input}""")

print("Please enter the website that you would like to use as context:")
# Get the website data
while True:
    try:
        website = input()
        loader = WebBaseLoader(website)
        docs = loader.load()
        break
    except:
        print("Invalid website. Please try again:")

# Create embedding model
embeddings = OllamaEmbeddings(model="llama3")

# Index the documents into vectorstore "vector"
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
# documents = text_splitter.split_text("I am jashan")
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()
# Create the chain
document_chain = create_stuff_documents_chain(llm, prompt) | output_parser
retrieval_chain = create_retrieval_chain(retriever, document_chain)

query = ""
while True:
    print("Please ask a question (type exit to exit the program)")
    query = input()
    if query == "exit":
        break
    response = document_chain.invoke({"input": query, "context": documents})
    print(response)