from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from typing import List
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


def ollama_query(context, question):
    template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model="qwen2.5:7b")
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"question": question, "context": context})

    # 过滤掉<think>...</think>部分

    # filtered_response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
    return response


def load_chroma_db(
    db_path="./rag_chroma",
    embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en"),
):
    chromadb = Chroma(
        embedding_function=embedding,
        persist_directory=db_path,
        collection_name="navie_rag",
    )
    return chromadb


def multi_query_search(query, db):

    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(),
        llm=OllamaLLM(model="qwen2.5:7b"),
        include_original=True,
    )
    print("retrieveing")
    return retriever.invoke(query)


def build_context(relevant_chunks: List[Document]) -> str:
    """
    Builds a context string from retrieved relevant document chunks.

    Parameters:
    relevant_chunks (List[Document]): A list of retrieved relevant document chunks.

    Returns:
    str: A concatenated string containing the content of the relevant chunks.
    """

    print("Context is built from relevant chunks")
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    print("Context is built from relevant chunks done")
    return context


def main():
    db = load_chroma_db()
    query = "What is LSTM and its architecture?"
    docs = multi_query_search(query, db)
    context = build_context(docs)
    # print(context)
    response = ollama_query(context, query)
    print("AI回答:", response)


if __name__ == "__main__":
    main()
