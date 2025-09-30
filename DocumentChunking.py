
import os
import hashlib
import traceback
import fitz
from typing import List,Dict,Tuple
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentChunking:
  def __init__(self, PDF_Path, chunk_size, chunk_overlap):
    self.PDF_Path = PDF_Path
    self.Chunk_Size = chunk_size
    self.Chunk_Overlap = chunk_overlap

  def PDF_Hashing(self):
    try:
      hasher = hashlib.md5()
      with open(self.PDF_Path, 'rb') as fl:
        while chunk:=fl.read(8192):
          hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as ex:
      print(f"Exception: {ex}")
      print(traceback.print_exc())
      raise ValueError(f"Exception to hash PDF {self.PDF_Path}: {ex}")


  def Chunking_Document(self):
    try:

      if not os.path.exists(self.PDF_Path):
        raise FileNotFoundError(f"Given file {self.PDF_Path} not found")
      hash_doc = self.PDF_Hashing()

      Recursive_text_splitter = RecursiveCharacterTextSplitter(
          chunk_size = self.Chunk_Size,
          chunk_overlap = self.Chunk_Overlap,
          length_function = len)


      doc_chunks = []
      with fitz.open(self.PDF_Path) as document:
        for idx, page in enumerate(document):
          texts_in_page = page.get_text()
          if texts_in_page.strip():
            page_chunks = Recursive_text_splitter.split_text(texts_in_page)
            for idx_chnk, chunk_content in enumerate(page_chunks):
              doc_chunks.append({
                "content":chunk_content,
                "metadata":{
                  "document_id":hash_doc,
                  "Source_File":os.path.basename(self.PDF_Path),
                  "Page_Number":idx+1,
                  "Chunk_ID": f"{hash_doc}-{idx}-{idx_chnk}" } })
        if not doc_chunks:
          raise ValueError(f"No text extracted from {self.PDF_Path}")
        else:
          return hash_doc, doc_chunks

    except Exception as ex:
      print(f"Exception occured in chunking: {ex}")
      print(traceback.print_exc())
      return None, []




