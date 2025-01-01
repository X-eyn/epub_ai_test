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
                model="gpt-4o",
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
    
    # The method name needs to match what we're calling
    def analyze_content(self, text: str) -> Dict:  # Changed from generate_summary
        """Generate comprehensive analysis of the text"""
        try:
            prompt = f"""Analyze this text and provide a detailed structured analysis.

            Text to analyze:
            {text}

            Provide a comprehensive analysis in this exact JSON format:
            {{
                "abstract": {{
                    "title": "Document Analysis",
                    "content": "Write a 2-3 paragraph summary of the main content and purpose of this document"
                }},
                "chapterSuggestions": [
                    {{
                        "title": "Suggest a relevant chapter title based on the content",
                        "keyPoints": ["Key point 1 from this section", "Key point 2 from this section"]
                    }}
                ],
                "studyQuestions": [
                    {{
                        "question": "Write a specific question about this document's content",
                        "type": "Conceptual/Technical/Analytical",
                        "suggestedAnswer": "Provide relevant answer points based on the document"
                    }}
                ],
                "keyInsights": [
                    "List specific, concrete insights from this document",
                    "Focus on actual content rather than generic points"
                ]
            }}
            
            Important: Base all analysis strictly on the actual content of the provided text."""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                response_format={ "type": "json_object" },
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            print("Analysis result:", result)  # Debug print
            return result
        except Exception as e:
            print(f"Error in content analysis: {str(e)}")
            return {
                "abstract": {
                    "title": "Error in Analysis",
                    "content": "Could not analyze the document content."
                },
                "chapterSuggestions": [],
                "studyQuestions": [],
                "keyInsights": []
            } 
            
def generate_summary(self, text: str) -> Dict:
    """Generate comprehensive analysis of the text"""
    try:
        # Truncate text if too long, but keep enough for good analysis
        max_length = 4000  # Adjust based on your needs
        text_for_analysis = text[:max_length]
        
        prompt = f"""Analyze this text and provide a detailed structured analysis.

        Text to analyze:
        {text_for_analysis}

        Provide a comprehensive analysis in this exact JSON format:
        {{
            "abstract": {{
                "title": "Document Analysis",
                "content": "Write a 2-3 paragraph summary of the main content and purpose of this document"
            }},
            "chapterSuggestions": [
                {{
                    "title": "Suggest a relevant chapter title based on the content",
                    "keyPoints": ["Key point 1 from this section", "Key point 2 from this section"]
                }}
            ],
            "studyQuestions": [
                {{
                    "question": "Write a specific question about this document's content",
                    "type": "Conceptual/Technical/Analytical",
                    "suggestedAnswer": "Provide relevant answer points based on the document"
                }}
            ],
            "keyInsights": [
                "List specific, concrete insights from this document",
                "Focus on actual content rather than generic points"
            ]
        }}
        
        Important: Base all analysis strictly on the actual content of the provided text.
        Be specific and reference actual content rather than making generic statements."""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user", 
                "content": prompt
            }],
            response_format={ "type": "json_object" },
            temperature=0.7,
            max_tokens=2000  # Adjust based on your needs
        )
        
        result = json.loads(response.choices[0].message.content)
        print("Analysis result:", result)  # Debug print
        return result
    except Exception as e:
        print(f"Error in content analysis: {str(e)}")
        return {
            "abstract": {
                "title": "Error in Analysis",
                "content": "Could not analyze the document content."
            },
            "chapterSuggestions": [],
            "studyQuestions": [],
            "keyInsights": []
        }               
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
        # First extract text from PDF
        processor = PDFProcessor()
        text = processor.extract_text(file)
        
        # Then analyze the extracted text
        analyzer = ContentAnalyzer()
        analysis = analyzer.analyze_content(text)  # Changed from generate_summary to analyze_content
        
        return analysis
    except Exception as e:
        print(f"Error analyzing content: {str(e)}")
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