from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import api_routes

app = FastAPI(title="ML Pipeline API")

# Configure CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_routes.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fullstack ML Platform API"}
