"""
firebase.py — Firebase Admin SDK initialization.
Initializes once and provides the Firestore client.
Credentials loaded from FIREBASE_CREDENTIALS env variable (path to service account JSON).
"""

import os
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

load_dotenv()

_db = None


def get_firestore() -> firestore.Client:
    global _db
    if _db is None:
        if not firebase_admin._apps:
            cred_path = os.getenv("FIREBASE_CREDENTIALS")
            if not cred_path:
                raise RuntimeError(
                    "FIREBASE_CREDENTIALS env variable is not set. "
                    "Set it to the path of your service account JSON file."
                )
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        _db = firestore.client()
    return _db
