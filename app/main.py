import os
from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import recommend, training
from app.services.cf_model import load_model
from app.services.gemini_service import load_metadata

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model
    print("Loading CF model...")
    load_model()
    await load_metadata()
    yield
    # Clean up the model and release the resources
    print("Successfully shut down")

app = FastAPI(
    title="Interior Design Furniture Recommendation API",
    description="AI-powered furniture recommendation using room image analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration - Allow connection from Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",           # Local development
        "http://localhost:3000",           # If Backend runs on 3000
        "http://localhost:5000",           # Common Backend port
        "https://capstone02.onrender.com", # Your Backend Render URL
        "*"                                # Allow all (for testing)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router, prefix="/api/v1", tags=["Recommendations"])
app.include_router(training.router, prefix="/api/v1/admin", tags=["Admin"])


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Interior Design Recommendation API is running",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Interior Design Recommendation API",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False  # Set to True only in development
    )