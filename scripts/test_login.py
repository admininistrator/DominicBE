from app.core.database import SessionLocal
from app.crud.crud_auth import verify_user_credentials

db = SessionLocal()
try:
    user = verify_user_credentials(db, 'test_user', '123456')
    if user:
        print(f"Login OK: username={user.username}, role={user.role}")
    else:
        print("FAILED: wrong password or user not found")
finally:
    db.close()

