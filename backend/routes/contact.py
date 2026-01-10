from fastapi import APIRouter
from database.db import contact_collection

router = APIRouter()

@router.post("/contact")
def save_message(email: str, message: str):
    contact_collection.insert_one({
        "email": email,
        "message": message
    })

    return {"status": "Message received successfully"}
