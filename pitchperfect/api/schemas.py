# pitchperfect/api/schemas.py
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class EmotionEnum(str, Enum):
    happy = "happy"
    sad = "sad"
    angry = "angry"
    fear = "fear"
    surprise = "surprise"
    disgust = "disgust"
    neutral = "neutral"


class SentimentEnum(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class SeverityEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


# Input schemas
class ProcessRequest(BaseModel):
    audio_path: str
    voice_sample_path: Optional[str] = None
    output_path: Optional[str] = None
    user_preferences: Optional[Dict] = Field(default_factory=dict)


# Output schemas
class TranscriptionOutput(BaseModel):
    text: str
    confidence: Optional[float] = None
    segments: Optional[List] = None


class SentimentOutput(BaseModel):
    emotion: EmotionEnum
    confidence: float = Field(ge=0.0, le=1.0)
    emotion_scores: Dict[str, float]
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    sentiment: SentimentEnum


class TonalOutput(BaseModel):
    prosodic_features: Dict
    voice_quality: Dict
    acoustic_problems: List[str]
    spectral_features: Dict


class ImprovementsOutput(BaseModel):
    improved_text: str
    original_text: str
    issues: Dict
    feedback: Dict
    prosody_guide: Dict
    ssml_markup: str


class ProcessResponse(BaseModel):
    timestamp: datetime
    transcription: TranscriptionOutput
    sentiment: SentimentOutput
    tonal: TonalOutput
    improvements: ImprovementsOutput
    synthesis: Optional[Dict] = None
    metrics: Dict
    error: Optional[str] = None
