from fastapi import APIRouter
from database.db import user_collection
import bcrypt

router = APIRouter()

@router.post("/register")
def register(email: str, password: str):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    user_collection.insert_one({
        "email": email,
        "password": hashed
    })
    return {"status": "User registered"}

@router.post("/login")
def login(email: str, password: str):
    user = user_collection.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode(), user["password"]):
        return {"status": "Login successful"}
    return {"status": "Invalid credentials"}
