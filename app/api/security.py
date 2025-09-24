import os
import jwt
import base64
import hashlib
from datetime import datetime, timedelta, timezone
from cryptography.fernet import Fernet

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY is not set in the environment. Please add it to your .env file."
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

hashed_secret = hashlib.sha256(JWT_SECRET_KEY.encode()).digest()

encryption_key = base64.urlsafe_b64encode(hashed_secret)
fernet = Fernet(encryption_key)


def create_mcp_access_token(galaxy_api_key: str) -> str:
    encrypted_api_key = fernet.encrypt(galaxy_api_key.encode()).decode()

    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {"exp": expire, "sub": "mcp_user", "galkey_enc": encrypted_api_key}

    encoded_jwt = jwt.encode(
        to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM
    )  # pyright: ignore[reportArgumentType]
    return encoded_jwt


def decrypt_api_key_from_token(token: str) -> str | None:

    try:
        payload = jwt.decode(
            token, JWT_SECRET_KEY, algorithms=[ALGORITHM]
        )  # pyright: ignore[reportArgumentType]
        encrypted_key = payload.get("galkey_enc")
        if not encrypted_key:
            raise ValueError("Token payload is missing the encrypted key.")

        decrypted_key = fernet.decrypt(encrypted_key.encode()).decode()
        return decrypted_key
    except (jwt.PyJWTError, ValueError) as e:
        print(f"Token validation/decryption failed: {e}")
        return None
