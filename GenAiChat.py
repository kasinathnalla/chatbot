import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

OPEN_API_KEY = "YOUR OWN OPEN API KEY"

st.header("My FirstChatbot")

st.sidebar.title("Your documents")
file = st.sidebar.file_uploader("Upload a PDF file and start asking questions", type="pdf")
if file:
    st.sidebar.success("File uploaded successfully!")
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        #        st.write(text)
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_API_KEY)

    # create a vector store FAISS

    vector_store = FAISS.from_texts(chunks, embeddings)
    user_question = st.text_input("Type your question here")

    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)
        llm = ChatOpenAI(
            openai_api_key=OPEN_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)
