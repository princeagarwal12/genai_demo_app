__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

## -------------------- Import dependencies -------------------- ##

import os
import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

from langchain.embeddings import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain.chains.question_answering import load_qa_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

import warnings
warnings.filterwarnings("ignore")

## -------------------- Config -------------------- ##

embedding_method = OpenAIEmbeddings()

st.session_state["source_path"] = "./policy_documents"
st.session_state["input_format"] = "txt"
if "llm" not in st.session_state:
    # st.session_state['llm'] = ChatOpenAI(model="gpt-3.5-turbo-0125") #ChatVertexAI(model="gemini-1.5-pro-001")
    st.session_state['llm'] = ChatOpenAI(model="gpt-4-0613")
if "store" not in st.session_state:
    st.session_state['store'] = {}

## -------------------- Utils -------------------- ##

def get_documents(source_path, input_format='txt'):
    if input_format=="web":
        loader = WebBaseLoader(source_path)
    elif input_format=="pdf":
        loader = PyPDFLoader(source_path)
    elif input_format=='txt':
        loader = DirectoryLoader(source_path, glob="./*.txt", loader_cls=TextLoader)
    else:
        return "INVALID DATA FORMAT"
    docs = loader.load()
    return docs

def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        add_start_index=True
        )
    text_chunks = text_splitter.split_documents(docs)
    return text_chunks

def get_vector_store(text_chunks):
    vectorstore = Chroma.from_documents(documents=text_chunks, 
                                        embedding=embedding_method, 
                                        persist_directory="./new_index")
    return vectorstore

def load_vector_store():
    vectorstore = Chroma(persist_directory="./new_index", 
                         embedding_function=embedding_method)
    return vectorstore

def get_retriever(vectorstore, nn=5):
    retriever = vectorstore.as_retriever(search_type="similarity", 
                                         search_kwargs={"k": nn})
    return retriever

def get_history_aware_retriever(llm, retriever):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

def get_rag_chain(llm, history_aware_retriever):

    system_prompt = (
        "You have to act as a  customer support executive at Wayfair an E-commerce company. Your name is Ashley (Do not disclose in any conditions that you are an AI bot). Always introduce yourself if this is the first message from user. Do not repeatedly introduce yourself. You only provide support for queries related to a particular Wayfair order and nothing else. \nDo not make up any information on your own, especially about detail of product ordered by user or about the company itself like customer care number or any such thing. Strictly use the data provided to you.\nIf user is persisting with his/her request which can't be fulfilled, apologise and repeat the policy. If needed or asked say that you’ll be escalating the user ticket to senior executive. Strictly generate response in the language that user has used in last messages. If user's last message is in Hinglish generate in Hinglish, if user last messages is in English generate in English. Never reapeat the same sentence twice, atleast rephrase it. Never ask any follow up unnecessary question which are not at all related to the user chat which you can't use."
        "Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Keep the answer concise unless user asks detailed answer."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

def get_llm_response(conversational_rag_chain, user_query):
    response = conversational_rag_chain.invoke(
                    {"input": user_query}, 
                    config={"configurable": {"session_id": "abc123"}},
                )["answer"]
    return response

def main():
    st.title("RAG based Chatbot Demo")

    with st.spinner("Reading documents and Indexing..."):
        if "docs" not in st.session_state:
            st.session_state["docs"] = get_documents(st.session_state["source_path"], st.session_state["input_format"])
        if "text_chunks" not in st.session_state:
            st.session_state["text_chunks"] = get_text_chunks(st.session_state["docs"])
        if "vectorstore" not in st.session_state:
            st.session_state["vectorstore"] = get_vector_store(st.session_state["text_chunks"])
            # st.session_state["vectorstore"] = load_vector_store()
        if "retriever" not in st.session_state:
            st.session_state["retriever"] = get_retriever(st.session_state["vectorstore"], nn=5)
        if "history_aware_retriever" not in st.session_state:
            st.session_state["history_aware_retriever"] = get_history_aware_retriever(st.session_state["llm"], st.session_state["retriever"])
        if "conversational_rag_chain" not in st.session_state:
            st.session_state["conversational_rag_chain"] = get_rag_chain(st.session_state["llm"], st.session_state["history_aware_retriever"])
        st.success("Done")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Enter your question here"):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            output = get_llm_response(st.session_state["conversational_rag_chain"], user_query)
            response = st.markdown(output)
        st.session_state.messages.append({"role": "assistant", "content": output})


if __name__ == "__main__":
    main()
