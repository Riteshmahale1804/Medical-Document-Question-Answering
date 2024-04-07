import streamlit as st
from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
import json
import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain.llms.ctransformers import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader

local_llm = "BioMistral-7B.Q4_K_M.gguf"


# Initialize LLM
llm = LlamaCpp(
    model_path=local_llm,
    temperature=0.3,
    max_tokens=2048,
    top_p=1
)

prompt_template = """Only return the helpful answer. Answer must be detailed and well explained.
Use the following pieces of information from the provided PDF to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
don't give answer out of the PDF or provided documents.

Context: {context}
Question: {question}


 
 

Only return the helpful answer. Answer must be detailed and well explained.
 
Helpful answer:

"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Declaring the prompt format
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
load_vector_store = Chroma(persist_directory="stores/doc_cosine", embedding_function=embeddings)

#initializing the retriever
retriever = load_vector_store.as_retriever(search_kwargs={"k": 1})


# Define the routes
def main():
    st.title('diabetes QA')

    query = st.text_input('Enter your question here:')
    if st.button('Get Response'):
        if query:
            # The logic to handle the query
            chain_type_kwargs = {"prompt": prompt}
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs=chain_type_kwargs,
                verbose=True
            )

            response = qa(query)
            # Extract the source document and the answer from the response
            if response['source_documents']:
                source_document = response['source_documents'][0].page_content
                doc = response['source_documents'][0].metadata['source']
            # If no source document is found, set the source document to "No source document found."
            else:
                source_document = "No source document found."
                doc = "Unknown"
            # Selecting the answer and source_doc etc. from the response
            answer = response['result']
            # Display the response
            st.write('Answer:', answer)
            st.write('Source Document:', source_document)
            # st.write('Document:', doc)
        else:
            st.warning('Please enter a question.')

if __name__ == '__main__':
    main()


