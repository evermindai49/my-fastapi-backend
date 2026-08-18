import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import AliasChoices, BaseModel, Field, ValidationError, model_validator

# ------------------------------------------------------------------------------
# Environment & Configuration Setup
# ------------------------------------------------------------------------------
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
    ENV_PATH = BASE_DIR / ".env"
    load_dotenv(dotenv_path=ENV_PATH)
except Exception as e:
    print(f"[INFO] Skipping local .env load in cloud runtime: {e}")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Defaulting to active, production-ready Groq model (llama-3.1-8b-instant)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    print("[WARNING] GROQ_API_KEY environment variable is missing!")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY or "missing_key",
)

app = FastAPI(
    title="EduTech & Skill-Up Groq-Powered API",
    version="1.4.1",
    description="AI learning backend using hosted Groq inference engine on Vercel.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Global Exception Handlers
# ------------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )


# ------------------------------------------------------------------------------
# Enums and Schemas
# ------------------------------------------------------------------------------
class DifficultyLevel(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"


class UserRole(str, Enum):
    STUDENT = "student"
    INSTRUCTOR = "instructor"
    ADMIN = "admin"


# --- Auth Schemas ---
class LoginRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None


class User(BaseModel):
    id: str
    email: str
    name: str
    role: UserRole


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User


# --- Skill Path Schemas ---
class SkillPathRequest(BaseModel):
    topic: str = "General Learning Path"
    difficulty: Optional[str] = "Beginner"
    goals: Optional[str] = ""

    @model_validator(mode="before")
    @classmethod
    def extract_and_normalize(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return {"topic": str(data)}

        topic_value = None
        for key in ["topic", "skill", "skill_name", "skill_path", "prompt", "course", "title", "query"]:
            if data.get(key):
                topic_value = str(data[key])
                break

        data["topic"] = topic_value or "General Learning Path"

        if "difficulty" in data:
            diff = data["difficulty"]
            if hasattr(diff, "value"):
                data["difficulty"] = str(diff.value)
            elif isinstance(diff, str):
                data["difficulty"] = diff
            else:
                data["difficulty"] = "Beginner"
        else:
            data["difficulty"] = "Beginner"

        if "goals" not in data or data["goals"] is None:
            data["goals"] = ""

        return data


class LessonItem(BaseModel):
    lesson_id: str = Field(..., validation_alias=AliasChoices("lesson_id", "id"))
    title: str
    duration: str = "45 mins"

    @model_validator(mode="before")
    @classmethod
    def coerce_id_string(cls, data: Any) -> Any:
        if isinstance(data, dict):
            lid = data.get("lesson_id") or data.get("id")
            if lid is not None:
                data["lesson_id"] = str(lid)
        return data


class ModuleItem(BaseModel):
    title: str
    description: str
    lessons: List[LessonItem]


class SkillPathResponse(BaseModel):
    title: str
    description: str
    modules: List[ModuleItem]


# --- Lesson Content Schemas ---
class LessonContentRequest(BaseModel):
    topic: str = Field(..., validation_alias=AliasChoices("topic", "lesson_title", "title", "subject"))
    module_title: Optional[str] = Field(None, validation_alias=AliasChoices("module_title", "module", "course"))
    course_name: Optional[str] = Field(None, validation_alias=AliasChoices("course_name", "subject_course", "course"))

    @model_validator(mode="before")
    @classmethod
    def normalize_lesson_request(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("topic"):
                for key in ["subject", "lesson_title", "title", "course"]:
                    if data.get(key):
                        data["topic"] = str(data[key])
                        break
        return data


class LessonContentResponse(BaseModel):
    title: str
    content: str
    key_takeaways: List[str]


# --- Exercise Schemas ---
class ExerciseRequest(BaseModel):
    topic: str = Field(..., validation_alias=AliasChoices("topic", "exercise_title", "title"))


class ExerciseResponse(BaseModel):
    title: str
    instructions: str
    initial_code: str = Field(..., validation_alias=AliasChoices("initial_code", "starter_code", "code"))
    hints: List[str]


# --- Evaluation Schemas ---
class SubmissionRequest(BaseModel):
    exercise_title: str = Field(..., validation_alias=AliasChoices("exercise_title", "title"))
    user_code: str = Field(..., validation_alias=AliasChoices("user_code", "submission", "code"))


class FeedbackResponse(BaseModel):
    is_correct: bool
    score: int
    feedback: str
    suggestions: List[str]


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def clean_json_response(raw_text: str) -> str:
    """Strips markdown code fences and isolates valid JSON strings."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace : last_brace + 1]

    return text.strip()


def generate_content_local(
    prompt: str,
    response_schema: Any,
    system_instruction: str,
    temperature: float = 0.2,
):
    """Generates structured content via Groq API endpoint."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GROQ_API_KEY environment variable is missing on Vercel environment settings.",
        )

    try:
        schema_json = json.dumps(response_schema.model_json_schema())
        enhanced_system_prompt = (
            f"{system_instruction}\n\n"
            f"CRITICAL REQUIREMENT: Output strictly valid JSON matching this JSON Schema:\n{schema_json}"
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )

        class LocalResponseWrapper:
            def __init__(self, content: str):
                self.text = content

        return LocalResponseWrapper(response.choices[0].message.content or "{}")

    except Exception as e:
        print(f"[ERROR] Groq API Execution Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Groq API call failed: {str(e)}",
        )


def parse_llm_json(raw_text: str, schema_class: Any):
    """Cleans raw text output and validates against target Pydantic schema."""
    cleaned_text = clean_json_response(raw_text)
    try:
        data = json.loads(cleaned_text)
        return schema_class.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"[ERROR] JSON Validation Error: {e}\nRaw Output:\n{raw_text}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model produced output that failed schema validation. Please retry.",
        )


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to EduTech API. Backend active on Vercel."}


@app.get("/health")
def health_check():
    return {"status": "ok", "provider": "Groq", "model": GROQ_MODEL}


@app.post("/api/v1/auth/login", response_model=AuthResponse)
def login_user(payload: LoginRequest):
    identifier = payload.username or payload.email or "student@example.com"
    return AuthResponse(
        access_token="dev_local_token_12345",
        token_type="bearer",
        user=User(
            id="usr_01",
            email=identifier,
            name=identifier.split("@")[0].title(),
            role=UserRole.STUDENT,
        ),
    )


@app.post("/api/v1/generate-path", response_model=SkillPathResponse)
def generate_skill_path(payload: SkillPathRequest):
    system_instruction = "You are an expert curriculum designer. Output strictly valid JSON."
    prompt = f"""
    Create a practical learning path curriculum for:
    - Topic: {payload.topic}
    - Level: {payload.difficulty}
    - Goals: {payload.goals or 'Master fundamental skills.'}

    Return 3 concise modules with 2-3 lessons each. Provide string lesson_ids.
    """
    res_wrapper = generate_content_local(prompt, SkillPathResponse, system_instruction)
    return parse_llm_json(res_wrapper.text, SkillPathResponse)


@app.post("/api/v1/generate-lesson", response_model=LessonContentResponse)
def generate_lesson_content(payload: LessonContentRequest):
    system_instruction = "You are an expert technical instructor. Output strictly valid JSON."
    course_context = f"Course: {payload.course_name}" if payload.course_name else "General"
    module_context = f"Module: {payload.module_title}" if payload.module_title else "Core Concept"

    prompt = f"""
    Write a clear, structured educational lesson for:
    - Subject: {payload.topic}
    - Context: {course_context} | {module_context}

    Format the 'content' field in Markdown with section headers: Overview, Core Mechanics, Code Example, and Best Practices.
    Provide 4 concise items in 'key_takeaways'.
    """
    res_wrapper = generate_content_local(prompt, LessonContentResponse, system_instruction, temperature=0.2)
    return parse_llm_json(res_wrapper.text, LessonContentResponse)


@app.post("/api/v1/generate-exercise", response_model=ExerciseResponse)
def generate_exercise(payload: ExerciseRequest):
    system_instruction = "You are a coding instructor creating hands-on exercises. Output strictly valid JSON."
    prompt = f"""
    Create a practical coding exercise for:
    - Topic: {payload.topic}

    Include clear instructions, starter code with placeholder comments ('initial_code'), and 2-3 hints.
    """
    res_wrapper = generate_content_local(prompt, ExerciseResponse, system_instruction)
    return parse_llm_json(res_wrapper.text, ExerciseResponse)


@app.post("/api/v1/submit-answer", response_model=FeedbackResponse)
def submit_answer(payload: SubmissionRequest):
    system_instruction = "You are an automated code evaluator. Output strictly valid JSON."
    prompt = f"""
    Evaluate this exercise submission:
    - Challenge Title: {payload.exercise_title}
    - Submitted Code:
    ```
    {payload.user_code}
    ```

    Assess correctness (boolean 'is_correct'), score (0-100), concise feedback, and actionable suggestions.
    """
    res_wrapper = generate_content_local(prompt, FeedbackResponse, system_instruction, temperature=0.0)
    return parse_llm_json(res_wrapper.text, FeedbackResponse)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)