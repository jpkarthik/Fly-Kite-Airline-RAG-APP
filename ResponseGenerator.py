
import os
from typing import List, Dict
from groq import Groq

class ResponseGenerator:
  def __init__(self, api_key, model_name):
    self.api_key = api_key
    self.Model_Name = model_name
    self.System_Message ="You are Retrival Auggumented Retreival Chat BOT agent for Fly Kite Airlies for HR Policy document"


  def Response_Genrator(self, prompt):
    try:
      client = Groq(api_key=self.api_key)
      completion = client.chat.completions.create(
          model=self.Model_Name,
          messages=[{"role":"system","content": self.System_Message},
                    {"role":"user","content":prompt}],
          temperature=0.1,max_completion_tokens=None,
          top_p=1,stream=True,stop=None)
      response = ""
      for chunk in completion:
        if chunk.choices:
          content = chunk.choices[0].delta.content or ""
          response += content
      return response

    except Exception as ex:
      print(f"Exception in Response Generator: {ex}")
      print(traceback.print_exc())
      return f"Unable to retrive data for query \n {ex}"
