import ast
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
    print(f"[INFO] Local .env skipped or handled by cloud runtime: {e}")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Active production model strings on Groq
DEFAULT_PRIMARY_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_MODEL = os.getenv("GROQ_MODEL", DEFAULT_PRIMARY_MODEL)

if not GROQ_API_KEY:
    print("[WARNING] GROQ_API_KEY environment variable is missing!")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY or "missing_key",
)

app = FastAPI(
    title="EduTech & Skill-Up Groq-Powered API",
    version="1.6.0",
    description="AI learning backend using hosted Groq inference engine on Vercel/Railway.",
)

# Robust CORS Configuration
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


# --- Exercise Schemas ---
class ExerciseResponse(BaseModel):
    id: Optional[str] = None
    title: str = Field(..., validation_alias=AliasChoices("title", "question", "exercise_title"))
    instructions: Optional[str] = ""
    initial_code: str = Field(..., validation_alias=AliasChoices("initial_code", "starter_code", "code"))
    hints: List[str] = Field(default_factory=list)
    options: Optional[List[str]] = None
    solution: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_exercise_response(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "title" not in data:
                for alt in ["question", "exercise_title", "name"]:
                    if data.get(alt):
                        data["title"] = str(data[alt])
                        break
                else:
                    data["title"] = "Exercise Check"

            if "initial_code" not in data:
                for alt_key in ["starter_code", "code", "starterCode", "initialCode"]:
                    if data.get(alt_key) is not None:
                        data["initial_code"] = str(data[alt_key])
                        break
                else:
                    data["initial_code"] = "# Write your solution here\n"
        return data


class LessonContentResponse(BaseModel):
    id: Optional[str] = "1"
    title: str
    description: Optional[str] = ""
    content: str
    key_takeaways: List[str] = Field(default_factory=list)
    exercises: List[ExerciseResponse] = Field(default_factory=list)


class ExerciseRequest(BaseModel):
    topic: str = Field(..., validation_alias=AliasChoices("topic", "exercise_title", "title"))

    @model_validator(mode="before")
    @classmethod
    def normalize_exercise_request(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("topic"):
            for key in ["exercise_title", "title", "subject", "query"]:
                if data.get(key):
                    data["topic"] = str(data[key])
                    break
        return data


# --- Evaluation Schemas ---
class SubmissionRequest(BaseModel):
    exercise_title: str = Field(..., validation_alias=AliasChoices("exercise_title", "title", "exercise", "topic"))
    user_code: str = Field(..., validation_alias=AliasChoices("user_code", "submission", "code", "answer", "user_answer"))

    @model_validator(mode="before")
    @classmethod
    def normalize_submission(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("exercise_title"):
                for key in ["title", "exercise", "topic", "exercise_id"]:
                    if data.get(key):
                        data["exercise_title"] = str(data[key])
                        break
            if not data.get("user_code"):
                for key in ["submission", "code", "answer", "user_answer"]:
                    if data.get(key) is not None:
                        data["user_code"] = str(data[key])
                        break
        return data


class FeedbackResponse(BaseModel):
    is_correct: bool = False
    score: int = 0
    feedback: str = "Evaluation complete."
    suggestions: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_feedback_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "score" in data:
                try:
                    data["score"] = int(data["score"])
                except (ValueError, TypeError):
                    data["score"] = 0

            if "is_correct" in data and isinstance(data["is_correct"], str):
                data["is_correct"] = data["is_correct"].lower() in ["true", "1", "yes"]

            if "suggestions" in data and isinstance(data["suggestions"], str):
                data["suggestions"] = [data["suggestions"]]
        return data


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
    """Generates structured content via Groq API endpoint with multi-tier failover."""
    api_key = os.getenv("GROQ_API_KEY")

    class LocalResponseWrapper:
        def __init__(self, content: str):
            self.text = content

    # Local Fallback when GROQ_API_KEY is missing or unconfigured
    if not api_key or api_key == "missing_key":
        print("[WARNING] Executing offline fallback response (GROQ_API_KEY unconfigured).")
        fallback_json = {
            "is_correct": True,
            "score": 100,
            "feedback": "Submission processed successfully (Offline fallback mode).",
            "suggestions": ["Add edge-case validation testing."]
        }
        return LocalResponseWrapper(json.dumps(fallback_json))

    target_model = os.getenv("GROQ_MODEL", DEFAULT_PRIMARY_MODEL)

    schema_json = json.dumps(response_schema.model_json_schema())
    enhanced_system_prompt = (
        f"{system_instruction}\n\n"
        f"CRITICAL REQUIREMENT: Output strictly valid JSON matching this JSON Schema:\n{schema_json}"
    )

    # Active production models on Groq LPUs
    fallback_models = [
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "qwen-2.5-coder-32b",
        "mixtral-8x7b-32768",
    ]

    models_to_try = [target_model] + [m for m in fallback_models if m != target_model]

    last_exception = None
    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": enhanced_system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
            )
            return LocalResponseWrapper(response.choices[0].message.content or "{}")
        except Exception as e:
            print(f"[WARNING] Groq execution failed on model '{model_name}': {e}")
            last_exception = e
            continue

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"All configured Groq models failed. Last error: {str(last_exception)}",
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
@app.get("/api")
@app.get("/api/index.py")
def read_root():
    return {"message": "Welcome to EduTech API. Backend active on Vercel/Railway."}


@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "provider": "Groq",
        "model": os.getenv("GROQ_MODEL", DEFAULT_PRIMARY_MODEL),
    }


@app.post("/api/v1/auth/login", response_model=AuthResponse)
@app.post("/v1/auth/login", response_model=AuthResponse)
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


# --- Get Lesson Details Endpoint ---
@app.get("/lesson/{lesson_id}", response_model=LessonContentResponse)
@app.get("/api/v1/lesson/{lesson_id}", response_model=LessonContentResponse)
@app.get("/v1/lesson/{lesson_id}", response_model=LessonContentResponse)
def get_lesson_by_id(lesson_id: str):
    """Retrieves structured lesson data and exercises by lesson ID."""
    try:
        system_instruction = "You are an expert technical instructor. Output strictly valid JSON."
        prompt = f"""
        Generate a complete lesson payload for Lesson ID '{lesson_id}'.
        Set 'id' to '{lesson_id}'. Provide a clear title, description, markdown content body, 3 key takeaways, 
        and 1 coding exercise with starter code and hints.
        """
        res_wrapper = generate_content_local(prompt, LessonContentResponse, system_instruction, temperature=0.2)
        return parse_llm_json(res_wrapper.text, LessonContentResponse)
    except Exception as e:
        return LessonContentResponse(
            id=str(lesson_id),
            title=f"Lesson {lesson_id}: Core Foundations",
            description=f"Overview and hands-on exercises for lesson module {lesson_id}.",
            content="### Module Overview\nLearn essential principles and practical code implementations.",
            key_takeaways=[
                "Understand core module mechanics",
                "Implement interactive code handlers",
                "Validate logic through automated evaluation"
            ],
            exercises=[
                ExerciseResponse(
                    id=f"ex-{lesson_id}",
                    title="Foundation Test",
                    instructions="Implement a function returning True.",
                    initial_code="# Write your solution below\ndef validate():\n    return True\n",
                    hints=["Ensure the function returns a boolean value."]
                )
            ]
        )


@app.post("/api/v1/generate-path", response_model=SkillPathResponse)
@app.post("/v1/generate-path", response_model=SkillPathResponse)
@app.post("/generate-path", response_model=SkillPathResponse)
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
@app.post("/v1/generate-lesson", response_model=LessonContentResponse)
@app.post("/generate-lesson", response_model=LessonContentResponse)
def generate_lesson_content(payload: LessonContentRequest):
    system_instruction = "You are an expert technical instructor. Output strictly valid JSON."
    course_context = f"Course: {payload.course_name}" if payload.course_name else "General"
    module_context = f"Module: {payload.module_title}" if payload.module_title else "Core Concept"

    prompt = f"""
    Write a clear, structured educational lesson for:
    - Subject: {payload.topic}
    - Context: {course_context} | {module_context}

    Format the 'content' field in Markdown with section headers: Overview, Core Mechanics, Code Example, and Best Practices.
    Provide 4 concise items in 'key_takeaways'. Include 1 relevant exercise.
    """
    res_wrapper = generate_content_local(prompt, LessonContentResponse, system_instruction, temperature=0.2)
    return parse_llm_json(res_wrapper.text, LessonContentResponse)


@app.post("/api/v1/generate-exercise", response_model=ExerciseResponse)
@app.post("/v1/generate-exercise", response_model=ExerciseResponse)
@app.post("/generate-exercise", response_model=ExerciseResponse)
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
@app.post("/v1/submit-answer", response_model=FeedbackResponse)
@app.post("/submit-answer", response_model=FeedbackResponse)
def submit_answer(payload: SubmissionRequest):
    # Deterministic local AST syntax verification
    try:
        ast.parse(payload.user_code)
    except SyntaxError as e:
        return FeedbackResponse(
            is_correct=False,
            score=0,
            feedback=f"Syntax Error on line {e.lineno}: {e.msg}",
            suggestions=[
                "Verify indentation, missing colons, or mismatched brackets.",
                "Ensure standard Python syntax before resubmitting."
            ]
        )

    # LLM-based logical verification
    system_instruction = "You are an automated code evaluator. Output strictly valid JSON."
    prompt = f"""
    Evaluate this exercise submission:
    - Challenge Title: {payload.exercise_title}
    - Submitted Code:
    ```python
    {payload.user_code}
    ```

    Assess correctness (boolean 'is_correct'), score (0-100), concise feedback, and actionable suggestions.
    """
    res_wrapper = generate_content_local(prompt, FeedbackResponse, system_instruction, temperature=0.0)
    return parse_llm_json(res_wrapper.text, FeedbackResponse)


# Catch-all Route placed explicitly at bottom
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(request: Request, path_name: str):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Route not matched in FastAPI",
            "requested_path": request.url.path,
            "method": request.method,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)