import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from openai import OpenAI
import chromadb
from typing import List, Dict
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize ChromaDB for vector storage
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="pdf_content")

class PDFProcessor:
    def __init__(self):
        self.chunks = []
    
    def extract_text(self, pdf_file: UploadFile) -> str:
        """Extract text from uploaded PDF"""
        try:
            pdf = PdfReader(pdf_file.file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks"""
        try:
            words = text.split()
            chunks = []
            current_chunk = []
            current_size = 0
            
            for word in words:
                current_chunk.append(word)
                current_size += len(word) + 1
                
                if current_size >= chunk_size:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_size = 0
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            self.chunks = chunks
            return chunks
        except Exception as e:
            raise Exception(f"Error chunking text: {str(e)}")

class MCQGenerator:
    def __init__(self):
        self.client = client
    
    def generate_mcqs(self, text: str, num_questions: int = 5) -> List[Dict]:
        """Generate MCQs from text using OpenAI API"""
        try:
            prompt = f"""Generate {num_questions} multiple choice questions from the following text. 
            Format the response as a JSON array with each question having the following structure:
            {{
                "question": "question text",
                "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
                "correct_answer": "A",
                "explanation": "explanation of correct answer"
            }}
            
            Text: {text}"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            
            mcqs = json.loads(response.choices[0].message.content)
            return mcqs.get('questions', [])
        except Exception as e:
            raise Exception(f"Error generating MCQs: {str(e)}")

class ContentAnalyzer:
    def __init__(self):
        self.client = client
    
    def generate_summary(self, text: str) -> Dict:
        """Generate summary and key points from text"""
        try:
            prompt = """Analyze the following text and provide:
            1. A concise summary (max 3 paragraphs)
            2. Key points (max 5)
            3. Main topics covered
            
            Format as JSON with keys: 'summary', 'key_points', 'topics'
            
            Text: """ + text
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            raise Exception(f"Error generating summary: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Welcome to EPub AI API",
        "endpoints": {
            "docs": "/docs",
            "upload_pdf": "/upload-pdf",
            "generate_mcqs": "/generate-mcqs",
            "analyze_content": "/analyze-content",
            "search": "/search"
        }
    }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        processor = PDFProcessor()
        text = processor.extract_text(file)
        chunks = processor.chunk_text(text)
        
        # Store chunks in ChromaDB for search
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{"source": file.filename, "chunk_id": i}],
                ids=[f"{file.filename}_chunk_{i}"]
            )
        
        return {
            "message": "PDF processed successfully",
            "num_chunks": len(chunks),
            "filename": file.filename
        }
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")  # Server-side logging
        return JSONResponse(
            status_code=500,
            content={
                "message": "Error processing PDF",
                "detail": str(e)
            }
        )

@app.post("/generate-mcqs")
async def generate_mcqs(file: UploadFile = File(...), num_questions: int = 5):
    try:
        processor = PDFProcessor()
        text = processor.extract_text(file)
        
        mcq_gen = MCQGenerator()
        mcqs = mcq_gen.generate_mcqs(text, num_questions)
        
        return {"mcqs": mcqs}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": "Error generating MCQs",
                "detail": str(e)
            }
        )

@app.post("/analyze-content")
async def analyze_content(file: UploadFile = File(...)):
    try:
        processor = PDFProcessor()
        text = processor.extract_text(file)
        
        analyzer = ContentAnalyzer()
        analysis = analyzer.generate_summary(text)
        
        return analysis
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": "Error analyzing content",
                "detail": str(e)
            }
        )

@app.get("/search")
async def search(query: str, num_results: int = 3):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=num_results
        )
        
        return results
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": "Error performing search",
                "detail": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)