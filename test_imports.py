# Test imports one by one to identify the problematic import
import sys
print("Python version:", sys.version)

try:
    from dotenv import load_dotenv
    print("✅ dotenv imported successfully")
except Exception as e:
    print(f"❌ dotenv import failed: {e}")

try:
    import pymongo
    print("✅ pymongo imported successfully")
except Exception as e:
    print(f"❌ pymongo import failed: {e}")

try:
    import certifi
    print("✅ certifi imported successfully")
except Exception as e:
    print(f"❌ certifi import failed: {e}")

try:
    from sensor.constant.database import DATABASE_NAME
    print("✅ database constants imported successfully")
except Exception as e:
    print(f"❌ database constants import failed: {e}")

try:
    from sensor.constant.env_variable import MONGODB_URL_KEY
    print("✅ env_variable constants imported successfully")
except Exception as e:
    print(f"❌ env_variable constants import failed: {e}") 