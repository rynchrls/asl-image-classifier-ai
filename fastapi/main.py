from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os

from inference import inference

# from hand_detector import hand_detect

# from create_dataset import save_to_dataset

CAPTURE_DIR = "capture_hands"
os.makedirs(CAPTURE_DIR, exist_ok=True)  # Ensure folder exists


class createUser(BaseModel):
    username: str
    email: str
    password: str
    age: Optional[int] = None


app = FastAPI(
    title="My FastAPI Project",
    version="1.0.0",
    description="A production-ready FastAPI application.",
    docs_url="/docs",
    redoc_url="/redoc",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # allow all headers
)


# ✅ Health check route
@app.get("/", tags=["Root"])
def root():
    return {"message": "Welcome to FastAPI!"}


@app.post("/asl")
async def asl(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Save the uploaded file into the folder
    if file.filename is None:
        return JSONResponse(
            status_code=400, content={"message": "No filename provided in upload."}
        )
    file_path = os.path.join(CAPTURE_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    result = inference(file_path)
    if result is None:
        return {"message": "The Sign Language is incorrect", "letter": result}
    return {"message": "Image processed successfully", "letter": result}


# # ✅ Optional root route
# @app.get("/")
# def read_root():
#     return {"message": "Welcome to FastAPI!"}
