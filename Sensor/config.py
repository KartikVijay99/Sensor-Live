from dataclasses import dataclass
import os
import pymongo
from urllib.parse import quote_plus

@dataclass
class EnvironmentVariable:
    mongo_db_url:str = os.getenv("MONGO_DB_URL")

env_var = EnvironmentVariable()

# Create MongoDB client with proper URL encoding
if env_var.mongo_db_url:
    # For now, let's use a hardcoded properly encoded URL
    # You should set this in your .env file
    username = quote_plus("KartikVijay")
    password = quote_plus("BBmq@02489")
    
    mongo_url = f"mongodb+srv://{username}:{password}@cluster0.rkivi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    mongo_client = pymongo.MongoClient(mongo_url)
else:
    # Fallback to a default connection or raise an error
    raise ValueError("MONGO_DB_URL environment variable is not set")