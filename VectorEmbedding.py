
import chromadb
import hashlib
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction

class VectorEmbedding:
  def __init__(self, model_name,chromadb_Path, collection_name):
    self.model = SentenceTransformer(model_name)
    self.client = chromadb.PersistentClient(path=chromadb_Path)
    self.Collection_name = collection_name
    self.collection = self._get_or_create_collection()

    print(f"Chroma DB Initialized at {chromadb_Path}, collection {collection_name}")

  def _get_or_create_collection(self):
    class ChormaEmbeddingFunction(EmbeddingFunction):
      def __init__(self, model):
        self.model = model

      def __call__(self,texts: List[str]):
        return self.model.encode(texts).tolist()

    return self.client.get_or_create_collection(
        name = self.Collection_name,
        embedding_function = ChormaEmbeddingFunction(self.model) )

  def Embedding_Generator(self,text):
    return self.model.encode(text).tolist()

  def Add_chunk_To_VectorDB(self, documents):
    try:
      if not documents:
        return []
      ids=[]
      contents=[]
      metadatas=[]
      for doc in documents:
        ids.append(doc['metadata']['Chunk_ID'])
        contents.append(doc['content'])
        metadatas.append(doc['metadata'])

      self.collection.add(documents=contents, metadatas=metadatas, ids=ids)
      return ids
    except Exception as ex:
      print(f"Execption in Add_chunk_To_VectorDB: {ex}")
      return []

  def get_document_hash(self,hash_document):
    hash_result = self.collection.get(where={"document_id":hash_document},limit=1)
    return hash_result['ids'] if hash_result and hash_result['ids'] else[]

  def search(self, input_query,k_val=3):
    try:
      input_query_embedding = self.model.encode(input_query).tolist()
      search_result = self.collection.query(
          query_embeddings = [input_query_embedding],
          n_results = k_val,
          include=['documents','metadatas','distances'] )

      document_retrived = []
      if search_result and search_result['documents'] and search_result['metadatas']:
        for indx in range(len(search_result['documents'][0])):
          document_content = search_result['documents'][0][indx]
          document_metadata = search_result['metadatas'][0][indx]
          document_distance = search_result['distances'][0][indx]

          document_retrived.append({
              "content":document_content,
              "metadata":document_metadata,
              "distance":document_distance })

        print(f"Retrived {len(document_retrived)} documents from chroma db")

      return document_retrived

    except Exception as ex:
      print(f"Exception in search: {ex}")
      print(traceback.print_exc())
      return []


