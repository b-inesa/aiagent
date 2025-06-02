from langchain_google_vertexai import ChatVertexAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os, json
from google.oauth2 import service_account

LLM_MODEL = "gemini-2.0-flash-001"

class LLM:
    def __init__(self):
        credentials = service_account.Credentials.from_service_account_info(json.loads(os.environ["CREDENTIALS_JSON"]))

        self.llm = ChatVertexAI(model = LLM_MODEL, credentials=credentials, temperature = 0.2)        
        self.memory = ConversationSummaryBufferMemory(llm=self.llm, memory_key="chat_history")

    def create_chain(self):
        template = """        
        Ти відповідаєш на питання абітурієнта про вступ на бакалаврат до Національного університету "Київський політехнічний інститут" на Факультет інформатики та обчислювальної техніки (ФІОТ).
        Твоя відповідь має бути чітко на основі даних контексту. Якщо ти не знайшов відповідь скажи про це та за потреби уточни питання.
        Відповідай коротко і не пиши ніякої додаткової інформації. Під час генерації відповіді враховуй історію розмови.
        Історія розмови: {chat_history}        
        Питання: {human_input}"""      

        promptllm = PromptTemplate(template=template, input_variables=["chat_history", "human_input"])      

        llm_chain = LLMChain(prompt=promptllm, llm=self.llm, memory=self.memory, verbose=True)
        
        return llm_chain