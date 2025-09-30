
import os
import sys
import argparse
import logging
import warnings
import json
from typing import List,Dict
from dotenv import load_dotenv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"]=""
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

try:
  base_path = os.path.abspath((os.path.dirname(__file__)))
except:
  base_path = os.path.join(os.getcwd())

print(f"Base_path {base_path}")
load_dotenv(os.path.join(os.getcwd(),'.env'))
groq_api_key = os.getenv('GROQ_API_KEY')
chunk_size=int(os.getenv('CHUNK_SIZE'))
chunk_overlap=int(os.getenv('CHUNK_OVERLAP'))
sentence_transformer_model = os.getenv('SENTENCE_TRANSFORMER_MODEL')
groq_ll_model=os.getenv('GROQ_LLM_MODEL')
pdf_path=os.path.join(base_path,os.getenv('PDF_FILE_NAME'))
chroma_db_path= os.path.join(base_path,os.getenv('CHROMA_DB_PATH'))
collection_name="FlyKite_Airlines"
hf_token=os.getenv('HF_TOKEN')

prompt_instruction = "\n Answer: Provide a clear responde to the Question. incorporating relevant details from the provided context. Include the metadata like document_id, source_file, page_number for each document used in your response"

print(f"PDF PATH: {pdf_path}")
print(f"chroma_db_path: {chroma_db_path}")


parser = argparse.ArgumentParser(description="To Run a Specific Job in pipeline")

parser.add_argument('--job', type=str, required=True,
                    choices=['rag-build','search','hosting'],
                    help='jobs to execute')
parser.add_argument('--raw',action='store_true',
                    help='Display raw retrieved documents in JSON format without feeding to LLM')

args = parser.parse_args()
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from DocumentChunking import DocumentChunking
from VectorEmbedding import VectorEmbedding
from ResponseGenerator import ResponseGenerator

vectorDBObj = VectorEmbedding(sentence_transformer_model,chroma_db_path,collection_name)

if args.job == 'rag-build':
  if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"File {pdf_path} not found")


  chunker = DocumentChunking(pdf_path,chunk_size,chunk_overlap)
  doc_hash_id = chunker.PDF_Hashing()
  if doc_hash_id and vectorDBObj.get_document_hash(doc_hash_id):
    print(f"Document {pdf_path} already chunked and embbeded in chroma vector db")
  else:
    doc_id, chunks = chunker.Chunking_Document()
    print(f"doc_id {doc_id}")
    print(f"Chunk: {chunks}")
    print(f"{len(chunks)}")
    if not chunks:
      print(f"No Chunks generated for the {pdf_path}")
    else:
      chunk_ids = vectorDBObj.Add_chunk_To_VectorDB(chunks)
      print(f"Added {len(chunk_ids)} chunks in chromaDB")


elif args.job == 'search':
  user_query = input("Enter your Query: ")
  retrived_documents = vectorDBObj.search(user_query)
  #print(retrived_documents)
  if not retrived_documents:
    print("No Document retrived")
  else:
    retrieved_json = [{ "document_number": indx+1,
            "content": doc['content'],
            "metadata":doc['metadata'],
            "distance":doc['distance'] }for indx, doc in enumerate(retrived_documents) ]
    if args.raw:
      print(f"\n Retrived Documents:\n")
      print(json.dumps(retrieved_json, indent=2))
    else:
      print(f"\n Retrived Documents:\n")
      print(json.dumps(retrieved_json, indent=2))
      context = "\n".join([doc['content'] for doc in retrived_documents])
      prompt = f"context: {context} \n\n Question: {user_query}"
      llm_obj = ResponseGenerator(groq_api_key, groq_ll_model)
      response = llm_obj.Response_Genrator(prompt)
      print(f"Generated Response..\n {response}")

elif args.job == 'hosting':
  from HostingInHuggingFace import HostingInHuggingFace
  Hosting_Obj = HostingInHuggingFace(base_path,hf_token)
  Hosting_Obj.ToRunPipeline()

