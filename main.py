from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from dotenv import load_dotenv
import fitz
import io
import os

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CLIENT_URL")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
model = ChatGroq(model=os.getenv("MODEL"), groq_api_key=os.getenv("GROQ_API_KEY"))
vector_db = None


class QueryRequest(BaseModel):
    query: str


@dynamic_prompt
def retrieve_context(request: ModelRequest):
    global vector_db
    if vector_db is None or vector_db.index.ntotal == 0:
        return "No PDF uploaded. Ask user to upload a PDF first."

    user_message = request.messages[-1].content
    docs = vector_db.similarity_search(user_message, k=3)
    context = "\\n\\n".join([d.page_content for d in docs])
    return f"Use this context from PDF: {context}"


agent = create_agent(
    model=model,
    middleware=[retrieve_context],
    checkpointer=InMemorySaver(),
)
config = {"configurable": {"thread_id": "1"}}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_db

    try:
        if not file or file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Please upload a PDF file")

        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        stream = io.BytesIO(content)
        pdf_doc = fitz.open(stream=stream, filetype="pdf")
        docs = []

        for i, page in enumerate(pdf_doc):
            text = page.get_text()
            if text.strip():
                docs.append(Document(page_content=text, metadata={"page": i + 1}))

        pdf_doc.close()

        if not docs:
            raise HTTPException(status_code=400, detail="No text found in PDF")

        chunks = text_splitter.split_documents(docs)
        vector_db = FAISS.from_documents(chunks, embeddings)

        return {"message": "File uploaded successfully", "pages": len(docs)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/query")
async def generate_answer(request: QueryRequest):
    try:
        if vector_db is None or vector_db.index.ntotal == 0:
            raise HTTPException(
                status_code=400, detail="Please upload a PDF file first"
            )

        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Please provide a question")

        ai_response = agent.invoke(
            {"messages": [{"role": "user", "content": request.query}]}, config
        )

        if not ai_response["messages"][-1].content:
            raise HTTPException(status_code=500, detail="No response generated")

        return {"answer": ai_response["messages"][-1].content}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating answer: {str(e)}"
        )


@app.get("/")
async def root():
    return {"message": "PDF ChatBot API", "status": "running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
