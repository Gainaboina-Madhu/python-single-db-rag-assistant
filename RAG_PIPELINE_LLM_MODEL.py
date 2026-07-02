'''
 - In this files we are going to Load the Data from web Page
 - Load data -> Chunks -> Embedded(openai) -> Vectordb
 - call the api (openai api key)
'''

import numpy as np
import langchain
import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

import dotenv
from dotenv import load_dotenv
load_dotenv()
os.environ['openai_api_key'] = os.getenv('OPEN_AI_KEY')


data_from_internet = WebBaseLoader('https://www.tpointtech.com/python-tutorial')

#print(data_from_internet.load())


# RAG STAGE - 2: Chunks

chunk_obj = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)

individual_doc = chunk_obj.split_documents(data_from_internet.load())

print(f'total number of documents {len(individual_doc)}')

# RAG STAGE - 3: Embeddings

openai_embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

all_vectors = []
for i in individual_doc:
    all_vectors.append(openai_embed_model.embed_query(i.page_content))


print(f'Checking how many documents are there after converting it into vectors {len(all_vectors)}')


# RAG STAGE - 4: VectorsDB(Chroma)

chr_db = Chroma.from_documents(individual_doc, openai_embed_model)

# check how to know the data is stored in database or not

sample_input = 'What is python'
result = chr_db.similarity_search(sample_input, k=3)


# OPEN AI LLM

openai_llm = ChatOpenAI(model='gpt-4o')

# giving prompt

prompt = ChatPromptTemplate.from_template(
    """
      Answer the queries from the below content:
      <context>
      {context}
      </context>

      Question: {input}
      """
)

document_chain = create_stuff_documents_chain(openai_llm, prompt)


response = document_chain.invoke({
    "input":sample_input,
    "context":result
}
)
print(response)