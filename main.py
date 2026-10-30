import os
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, String, Integer, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, sessionmaker, Session, relationship
from pwdlib import PasswordHash
from jose import jwt, JWTError
from openai import AsyncOpenAI

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./aurix.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {"sslmode":"require"}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase): pass
def utcnow(): return datetime.now(timezone.utc)

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    memory_consent: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

class Memory(Base):
    __tablename__ = "memories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    category: Mapped[str] = mapped_column(String(60), default="general")
    content: Mapped[str] = mapped_column(Text)

app = FastAPI(title="AURIX AI API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["GET","POST","DELETE","OPTIONS"], allow_headers=["*"])
passwords = PasswordHash.recommended()
oauth = OAuth2PasswordBearer(tokenUrl="/v1/auth/login")
JWT_SECRET = os.getenv("JWT_SECRET", "")
if not JWT_SECRET:
    raise RuntimeError("Set JWT_SECRET in hosting environment variables")

def db_session():
    db=SessionLocal()
    try: yield db
    finally: db.close()

def token_for(user_id:int):
    exp=datetime.now(timezone.utc)+timedelta(minutes=30)
    return jwt.encode({"sub":str(user_id),"exp":exp},JWT_SECRET,algorithm="HS256")

def current_user(token:str=Depends(oauth), db:Session=Depends(db_session)):
    try:
        payload=jwt.decode(token,JWT_SECRET,algorithms=["HS256"])
        uid=int(payload["sub"])
    except (JWTError,ValueError,KeyError,TypeError):
        raise HTTPException(401,"Invalid or expired token")
    user=db.get(User,uid)
    if not user: raise HTTPException(401,"User not found")
    return user

class Credentials(BaseModel):
    email: EmailStr
    password: str=Field(min_length=10,max_length=128)
class ChatInput(BaseModel):
    message: str=Field(min_length=1,max_length=12000)
    language: str="hi"
    provider: str="openai"
class ConsentInput(BaseModel):
    enabled: bool
class MemoryInput(BaseModel):
    content: str=Field(min_length=1,max_length=5000)
    category: str="general"

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

@app.get("/")
def root(): return {"service":"AURIX AI API","health":"/health","docs":"/docs"}
@app.get("/health")
def health(): return {"status":"ok","service":"aurix-api"}

@app.post("/v1/auth/register")
def register(data:Credentials, db:Session=Depends(db_session)):
    email=data.email.lower()
    if db.query(User).filter(User.email==email).first():
        raise HTTPException(409,"Email already registered")
    user=User(email=email,password_hash=passwords.hash(data.password))
    db.add(user); db.commit(); db.refresh(user)
    return {"access_token":token_for(user.id),"token_type":"bearer","expires_in":1800}

@app.post("/v1/auth/login")
def login(data:Credentials, db:Session=Depends(db_session)):
    user=db.query(User).filter(User.email==data.email.lower()).first()
    if not user or not passwords.verify(data.password,user.password_hash):
        raise HTTPException(401,"Incorrect email or password")
    return {"access_token":token_for(user.id),"token_type":"bearer","expires_in":1800}

@app.post("/v1/chat")
async def chat(data:ChatInput, user:User=Depends(current_user)):
    provider=data.provider.lower()
    if provider=="openai":
        key=os.getenv("OPENAI_API_KEY")
        if not key: raise HTTPException(503,"OPENAI_API_KEY is not configured on server")
        client=AsyncOpenAI(api_key=key)
        system="You are AURIX, a helpful AI assistant. Reply in Hindi." if data.language=="hi" else "You are AURIX, a helpful AI assistant. Reply in English."
        try:
            result=await client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL","gpt-4.1-mini"),
                messages=[{"role":"system","content":system},{"role":"user","content":data.message}]
            )
            return {"reply":result.choices[0].message.content or "","provider":"openai"}
        except Exception:
            raise HTTPException(502,"AI provider request failed")
    raise HTTPException(503,"This deployment currently supports OpenAI chat only. Configure other providers in a later release.")

@app.post("/v1/memory/consent")
def memory_consent(data:ConsentInput,user:User=Depends(current_user),db:Session=Depends(db_session)):
    user.memory_consent=data.enabled; db.commit()
    return {"enabled":user.memory_consent}

@app.get("/v1/memory/consent")
def get_memory_consent(user:User=Depends(current_user)):
    return {"enabled":user.memory_consent}

@app.get("/v1/memory")
def list_memory(user:User=Depends(current_user),db:Session=Depends(db_session)):
    if not user.memory_consent: raise HTTPException(403,"Enable memory consent first")
    return [{"id":m.id,"content":m.content,"category":m.category} for m in db.query(Memory).filter(Memory.user_id==user.id).all()]

@app.post("/v1/memory")
def save_memory(data:MemoryInput,user:User=Depends(current_user),db:Session=Depends(db_session)):
    if not user.memory_consent: raise HTTPException(403,"Enable memory consent first")
    item=Memory(user_id=user.id,content=data.content,category=data.category)
    db.add(item); db.commit(); db.refresh(item)
    return {"id":item.id,"content":item.content,"category":item.category}

@app.delete("/v1/memory/{memory_id}")
def delete_memory(memory_id:int,user:User=Depends(current_user),db:Session=Depends(db_session)):
    item=db.query(Memory).filter(Memory.id==memory_id,Memory.user_id==user.id).first()
    if not item: raise HTTPException(404,"Memory not found")
    db.delete(item); db.commit()
    return {"deleted":True}
