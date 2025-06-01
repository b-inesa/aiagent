import PyPDF2, os, logging, PyPDF2.errors, re, asyncio
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

DOCUMENTS_FOLDER = "documents"
EMBEDDING_MODEL = "text-multilingual-embedding-002"

class Data:
    def __init__(self):                
        self.documents = []
        self.vectors = None

    def clean_pdf_text(self, text):
        text = re.sub(r'\n{2,}', '\n', text)  
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)      
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()
    
    def get_pdf_data(self):        
        all_text = []
        for file_name in os.listdir(DOCUMENTS_FOLDER):
            file_path = os.path.join(DOCUMENTS_FOLDER, file_name)
            try:
                with open(file_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:          
                        text = page.extract_text()
                        cleaned_text = self.clean_pdf_text(text)                         
                        all_text.append(cleaned_text)    
            except FileNotFoundError:
                logging.error(f"Файл не знайдено: {file_path}")
            except PyPDF2.errors.PdfReadError as e:
                logging.error(f"Помилка читання файлу {file_path}: {e}")
            except Exception as e:
                logging.error(f"Виникла помилка при обробці файлу {file_path}: {e}")

        total_text = "\n".join(all_text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n", ".", " "]) 
        self.documents = text_splitter.split_text(total_text)
    
    def vectorize_documents(self):
        embeddings = VertexAIEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectors = FAISS.from_texts(self.documents, embeddings) 

    async def get_context(self, question):   
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self.vectors.similarity_search, question, 5)
        doc_context = "\nКонтекст: " + "\n".join(result.page_content for result in results)
        logging.info(f"Знайдено контекст для питання: {question} \n{doc_context}")        
        return doc_context
