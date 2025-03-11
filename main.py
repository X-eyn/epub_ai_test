import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from openai import OpenAI
import chromadb
from typing import List, Dict
import json
from datetime import datetime
import magic
import langdetect
import tiktoken

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

# --- Token & Cost Estimation ---
class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.log_file = "token_cost_log.json"
        self.summary_file = "usage_summary.json"
        self.pricing = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "input_processing": {"input": 0.0, "output": 0.0}  # Added for file processing
        }

    def update_usage(self, model: str, input_tokens: int, output_tokens: int, file_name: str = "", input_text: str = "", file_size: int = 0):
        """Updates token usage, cost, and file details."""
        if model not in self.pricing:
            raise ValueError(f"Unknown model: {model}")

        input_cost = (input_tokens / 1000) * self.pricing[model]["input"]
        output_cost = (output_tokens / 1000) * self.pricing[model]["output"]
        total_cost = input_cost + output_cost

        self.total_tokens += input_tokens + output_tokens
        self.total_cost += total_cost

        file_type = ""
        file_extension = ""
        if file_name:
            try:
                file_type = magic.from_file(file_name, mime=True)
                file_extension = os.path.splitext(file_name)[1]
            except Exception as e:
                print(f"Error detecting file type/extension: {e}")

        word_count = len(input_text.split())
        char_count = len(input_text)

        detected_language = ""
        try:
            detected_language = langdetect.detect(input_text)
        except Exception as e:
            print(f"Error detecting language: {e}")

        # Log usage (detailed log)
        self.log_usage(
            model,
            input_tokens,
            output_tokens,
            total_cost,
            file_name,
            file_type,
            file_extension,
            file_size,
            word_count,
            char_count,
            detected_language
        )

        # Update summary data
        self.update_summary(
            file_name,
            file_type,
            file_extension,
            word_count,
            char_count,
            total_cost,
            model,
            input_tokens,
            output_tokens,
            detected_language
        )

    def get_usage_summary(self):
        """Returns the summarized usage data along with detailed logs."""
        try:
            with open(self.summary_file, "r") as f:
                summary_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            summary_data = {}

        try:
            with open(self.log_file, "r") as f:
                detailed_logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            detailed_logs = []

        return {
            "summary": summary_data,
            "details": detailed_logs
        }

    def log_usage(self, model: str, input_tokens: int, output_tokens: int, cost: float, file_name: str, file_type: str, file_extension: str, file_size: int, word_count: int, char_count: int, detected_language: str):
        """Logs usage details to a JSON file."""
        try:
            with open(self.log_file, "r") as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []

        logs.append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "file_name": file_name,
            "file_type": file_type,
            "file_extension": file_extension,
            "file_size_bytes": file_size,
            "word_count": word_count,
            "char_count": char_count,
            "detected_language": detected_language
        })

        with open(self.log_file, "w") as f:
            json.dump(logs, f, indent=4)

    def update_summary(self, file_name: str, file_type: str, file_extension: str, word_count: int, char_count: int, cost: float, model: str, input_tokens: int, output_tokens: int, detected_language: str):
        """Updates the summarized usage data JSON file."""
        try:
            with open(self.summary_file, "r") as f:
                summary_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            summary_data = {}

        if file_name not in summary_data:
            summary_data[file_name] = {
                "file_type": file_type,
                "file_extension": file_extension,
                "word_count": word_count,
                "char_count": char_count,
                "total_cost": cost,
                "usage_details": []
            }
        else:
            summary_data[file_name]["word_count"] += word_count
            summary_data[file_name]["char_count"] += char_count
            summary_data[file_name]["total_cost"] += cost

        # Add usage details to the summary
        summary_data[file_name]["usage_details"].append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "detected_language": detected_language
        })

        with open(self.summary_file, "w") as f:
            json.dump(summary_data, f, indent=4)

token_tracker = TokenTracker()

def get_token_count(text: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # For newer models like gpt-4o that don't have automatic mapping
        if model.startswith("gpt-4o"):
            # Use cl100k_base encoding which is used by GPT-4 models
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Default to cl100k_base for other unmapped models
            encoding = tiktoken.get_encoding("cl100k_base")
    
    num_tokens = len(encoding.encode(text))
    return num_tokens

class PDFProcessor:
    def __init__(self):
        self.chunks = []
    
    def extract_text(self, file: UploadFile) -> str:
        """Extract text from uploaded PDF"""
        try:
            pdf = PdfReader(file.file)
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
        model = "gpt-4o"
        file_name = "unknown"
        try:
            if os.path.isfile(text):
                file_name = os.path.basename(text)
                with open(text, 'r', encoding='utf-8') as f:
                    text = f.read()

            prompt = f"""Generate {num_questions} multiple choice questions from the following text. 
            Format the response as a JSON array with each question having the following structure:
            {{
                "question": "question text",
                "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
                "correct_answer": "A",
                "explanation": "explanation of correct answer"
            }}
            
            Text: {text}"""
            
            prompt_tokens = get_token_count(prompt, model)

            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            
            response_content = response.choices[0].message.content
            response_tokens = get_token_count(response_content, model)

            token_tracker.update_usage(
                model, 
                prompt_tokens, 
                response_tokens, 
                file_name=file_name, 
                input_text=text
            )

            mcqs = json.loads(response_content)
            return mcqs.get('questions', [])
        except Exception as e:
            raise Exception(f"Error generating MCQs: {str(e)}")

class ContentAnalyzer:
    def __init__(self):
        self.client = client

    def analyze_content(self, text: str, file_name: str) -> Dict:
        """Generate comprehensive analysis of the text"""
        model = "gpt-4o"
        try:
            if os.path.isfile(text):
                file_name = os.path.basename(text)
                with open(text, 'r', encoding='utf-8') as f:
                    text = f.read()

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

            prompt_tokens = get_token_count(prompt, model)

            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" },
                temperature=0.7
            )

            response_content = response.choices[0].message.content
            response_tokens = get_token_count(response_content, model)

            token_tracker.update_usage(
                model, 
                prompt_tokens, 
                response_tokens, 
                file_name=file_name, 
                input_text=text
            )

            return json.loads(response_content)
        except Exception as e:
            print(f"Error in content analysis: {str(e)}")
            return {
                "error": {
                    "message": "Error in content analysis",
                    "detail": str(e)
                }
            }

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Welcome to EPub AI API",
        "endpoints": {
            "docs": "/docs",
            "upload_pdf": "/upload-pdf",
            "generate_mcqs": "/generate-mcqs",
            "analyze_content": "/analyze-content",
            "search": "/search",
            "usage_summary": "/usage-summary"
        }
    }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded file
        with open(file.filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        # Reset file pointer for PDF processing
        file.file.seek(0)
        
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

        # Get file stats
        file_size = os.path.getsize(file.filename)
        
        # Update token tracker with minimal token counting
        input_tokens = len(text.split())  # Simple word count as token estimate
        token_tracker.update_usage(
            model="input_processing",
            input_tokens=input_tokens,
            output_tokens=0,
            file_name=file.filename,
            input_text=text,
            file_size=file_size
        )
        
        # Clean up the temporary file
        os.remove(file.filename)
        
        return {
            "message": "PDF processed successfully",
            "num_chunks": len(chunks),
            "filename": file.filename,
            "file_size_bytes": file_size,
            "estimated_tokens": input_tokens
        }
        
    except Exception as e:
        # Clean up the temporary file if it exists
        if os.path.exists(file.filename):
            os.remove(file.filename)
            
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
        # Create a temporary file and process it
        with open(file.filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Reset file pointer for PDF processing
        file.file.seek(0)
        
        processor = PDFProcessor()
        text = processor.extract_text(file)
        
        mcq_gen = MCQGenerator()
        mcqs = mcq_gen.generate_mcqs(text, num_questions)
        
        # Clean up temporary file
        os.remove(file.filename)
        
        return {"mcqs": mcqs}
    except Exception as e:
        # Clean up the temporary file if it exists
        if os.path.exists(file.filename):
            os.remove(file.filename)
            
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
        # Create a temporary file and process it
        with open(file.filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        # Reset file pointer for PDF processing
        file.file.seek(0)
        
        processor = PDFProcessor()
        text = processor.extract_text(file)
        
        # Analyze the extracted text
        analyzer = ContentAnalyzer()
        analysis = analyzer.analyze_content(text, file_name=file.filename)
        
        # Clean up temporary file
        os.remove(file.filename)
        
        return analysis
    except Exception as e:
        # Clean up the temporary file if it exists
        if os.path.exists(file.filename):
            os.remove(file.filename)
            
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

@app.get("/usage-summary")
async def usage_summary():
    """Provides summarized usage information and detailed logs."""
    return token_tracker.get_usage_summary()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)