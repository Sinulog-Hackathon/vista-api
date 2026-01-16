import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firebase():
    # Check if Firebase is already initialized to prevent errors during hot-reloads
    if not firebase_admin._apps:
        # Load the credentials
        cred = credentials.Certificate("./vista-37a9b-firebase-adminsdk-fbsvc-fa92788e85.json")
        
        # Initialize the app
        firebase_admin.initialize_app(cred)

# Initialize it immediately
initialize_firebase()

# Expose the Firestore client
db = firestore.client(database_id='vista-db')