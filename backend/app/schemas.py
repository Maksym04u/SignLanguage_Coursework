from typing import List, Optional

from pydantic import BaseModel, EmailStr


class UserSignup(BaseModel):
    email: EmailStr
    password: str
    preferred_language: str = "en"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    email: str | None = None


class UserMe(BaseModel):
    email: str
    preferred_language: str


class TranslationRequest(BaseModel):
    source_language: str = "en"
    tokens: List[str]


class TranslationResponse(BaseModel):
    raw_text: str
    corrected_text: Optional[str]


class TextToSignRequest(BaseModel):
    source_language: str = "en"
    text: str


class GestureSequenceFrame(BaseModel):
    lh: List[float]
    rh: List[float]


class GestureFrame(BaseModel):
    type: str  # "word" | "letter" | "space" | "missing" | "silent"
    label: str
    gloss: Optional[str] = None
    class_id: Optional[str] = None
    language: Optional[str] = None
    lh: Optional[List[float]] = None
    rh: Optional[List[float]] = None
    sequence: Optional[List[GestureSequenceFrame]] = None
    playback_format: Optional[str] = None  # e.g. "raw_v1"
    reason: Optional[str] = None


class TextToSignSummary(BaseModel):
    input_tokens: int
    words_matched: int
    letters_matched: int
    letters_missing: List[str]


class TextToSignResponse(BaseModel):
    language: str
    frames: List[GestureFrame]
    summary: TextToSignSummary


class PredictRequest(BaseModel):
    keypoints: List[List[float]]
    source_language: str = "en"


class PredictResponse(BaseModel):
    class_id: str
    language: str
    display_text: str
    confidence: float
