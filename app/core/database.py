from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from pathlib import Path
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Always resolve .env from project root: <repo>/.env
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

DB_USER = os.getenv("DB_USER", "dominic")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "chatbot_db")
DB_SSL = os.getenv("DB_SSL", "")  # set to "true" on Azure

# Sử dụng pymysql làm driver kết nối MySQL
encoded_password = quote_plus(DB_PASSWORD)
SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

connect_args: dict = {
    "connect_timeout": 10,   # fail fast (seconds) instead of hanging
    "read_timeout": 30,
    "write_timeout": 30,
}
if DB_SSL.lower() in ("true", "1", "yes"):
    connect_args["ssl"] = {"ssl_mode": "REQUIRED"}

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_timeout=10,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
