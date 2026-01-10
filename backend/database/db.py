from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["synapse"]
contact_collection = db["contacts"]
user_collection = db["users"]
