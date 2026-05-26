from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .db import get_db
from .models import User
from .schemas import (
    PredictRequest,
    PredictResponse,
    TextToSignRequest,
    TextToSignResponse,
    TokenResponse,
    TranslationRequest,
    TranslationResponse,
    UserLogin,
    UserMe,
    UserSignup,
)
from .security import create_access_token, decode_token, hash_password, verify_password
from .services.grammar import GrammarService
from .services.sign_model import sign_model_service
from .services.text_to_gesture import text_to_gesture_service
from .services.translator import TranslatorService

app = FastAPI(title="Multilingual Sign Translator API")
grammar_service = GrammarService()
translator_service = TranslatorService()
bearer_scheme = HTTPBearer(auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> int:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = credentials.credentials
    try:
        payload = decode_token(token)
        return int(payload["sub"])
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/auth/signup", response_model=TokenResponse)
def signup(data: UserSignup, db: Session = Depends(get_db)):
    user = User(
        email=data.email,
        password_hash=hash_password(data.password),
        preferred_language=data.preferred_language,
    )
    try:
        db.add(user)
        db.commit()
        db.refresh(user)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already registered") from exc
    return TokenResponse(
        access_token=create_access_token(str(user.id)),
        email=user.email,
    )


@app.post("/auth/login", response_model=TokenResponse)
def login(data: UserLogin, db: Session = Depends(get_db)):
    row = db.query(User).filter(User.email == data.email).first()
    if not row or not verify_password(data.password, row.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(
        access_token=create_access_token(str(row.id)),
        email=row.email,
    )


@app.get("/auth/me", response_model=UserMe)
def me(user_id: int = Depends(get_current_user_id), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserMe(email=user.email, preferred_language=user.preferred_language)


@app.post("/translate/sign-to-text", response_model=TranslationResponse)
def translate_sign_to_text(
    payload: TranslationRequest,
    _user_id: int = Depends(get_current_user_id),
):
    raw_text = translator_service.compose_text(payload.tokens)
    corrected_text = grammar_service.correct(raw_text, payload.source_language)
    return TranslationResponse(raw_text=raw_text, corrected_text=corrected_text)


@app.post("/translate/text-to-sign", response_model=TextToSignResponse)
def translate_text_to_sign(payload: TextToSignRequest, user_id: int = Depends(get_current_user_id)):
    result = text_to_gesture_service.translate(payload.text, payload.source_language)
    return TextToSignResponse(**result)


@app.post("/translate/predict", response_model=PredictResponse)
def predict_sign(payload: PredictRequest, user_id: int = Depends(get_current_user_id)):
    try:
        result = sign_model_service.predict(
            payload.keypoints,
            source_language=payload.source_language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model assets missing: {exc}") from exc
    return PredictResponse(**result)
