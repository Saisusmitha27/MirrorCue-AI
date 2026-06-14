## MirrorCue AI

A Bias-Aware Multi-Agent Resume Intelligence System

## Problem Statement

Many students submit resumes that fail Applicant Tracking System (ATS) screening or are not effectively tailored to specific job descriptions. As a result, qualified candidates are often filtered out before reaching recruiters, despite possessing relevant skills and experience.

Existing resume optimization tools primarily focus on keyword matching and generic rewriting while overlooking recruiter-side bias, contextual resume quality, and factual validation. Candidates receive limited actionable feedback and often struggle to improve resume visibility without introducing exaggerated or fabricated claims.

There is a need for an intelligent system that can evaluate resumes against job descriptions, identify ATS and content gaps, detect potential bias signals, and provide verified, actionable recommendations for improvement.

## Solution Overview

MirrorCue AI is a Bias-Aware Resume Intelligence System designed to help students optimize resumes for both ATS systems and human recruiters.

The platform uses a six-agent architecture to analyze resumes against job descriptions, identify keyword and skill gaps, evaluate ATS compatibility, detect recruiter-side bias indicators, and generate verified resume improvements.

Unlike traditional resume analyzers, MirrorCue AI introduces two key innovations: a Bias Mirror module that highlights candidate visibility risks and a Q&A-driven validation workflow that prevents hallucinated resume content. Every recommendation is generated from validated candidate information, ensuring that rewritten content remains accurate, defensible, and job-relevant.

The system produces three actionable outputs in a single workflow:

- ATS Analysis Report
- Bias Mirror Report
- Verified Resume Rewrite

This enables candidates to improve resume visibility, align with target roles, and present their experience more effectively without compromising authenticity.

## Key Features

* **Multi-Agent Orchestration**: Sequential, state-controlled analysis pipeline modeled via LangGraph.
* **Semantic & Priority-Based ATS Scoring**: Hybrid matching using sentence embeddings and priority-ordered fuzzy text matching.
* **Demographic Bias Mitigation**: XGBoost-powered classifier analyzing prestige, gender, regional, and career gap markers.
* **Interactive Smart Rewrite**: Automated generation of contextual resume enhancements gated by verification questions.
* **React Dashboard**: Modern web interface displaying real-time analysis logs, scoring breakdowns, and side-by-side bullet diffs.

## System Architecture

* **Frontend**: A single-page React app built with TypeScript and Vite. It utilizes Zustand for global state management and Tailwind CSS for interface styling, communicating with the backend via REST endpoints.
* **Backend**: An asynchronous FastAPI service that exposes endpoints for authentication, uploads, and agent analysis orchestration.
* **Database**: PostgreSQL (or Supabase) managed through SQLAlchemy async sessions. Dynamic analysis metadata, scoring details, and response blocks are stored in a schema using optimized `JSONB` columns.
* **AI Modules**: A LangGraph orchestrator coordinating an LLM-based Resume Parser, an ATS Matcher, a Bias Mirror (integrating a local XGBoost model), a Q&A Generator, and a Resume Rewrite Agent.
* **APIs**: JWT-secured endpoints for session management, PDF uploads, polling statuses, and submitting Q&A responses.
* **Internal Communication**: The state is passed sequentially across agents using a centralized state graph dictionary, preserving the context through processing stages.

## Tech Stack

### Frontend
* React 18 & Vite
* TypeScript
* Zustand (State Management)
* Tailwind CSS

### Backend
* FastAPI (Asynchronous Web Framework)
* SQLAlchemy (Async ORM)
* Alembic (Migrations)
* Uvicorn (ASGI Web Server)

### Database
* PostgreSQL (with JSONB support)(Supabase)

### AI / ML
* Google Gemini API (with Groq fallback)
* SentenceTransformers (`all-MiniLM-L6-v2`)
* XGBoost & Scikit-learn (Bias Classification)

## Project Structure

```text
MirrorCueAI/
├── backend/
│   ├── agents/          # LangGraph orchestrator and specialized LLM agents
│   ├── core/            # Config settings, logging definitions, and DB engines
│   ├── data/            # Static bias dictionaries and training datasets
│   ├── ml/              # Feature extractors and XGBoost classification code
│   ├── models/          # SQLAlchemy async DB tables (User, Resume, Analysis)
│   ├── routers/         # REST API endpoints (Auth, Resume, Analysis)
│   ├── schemas/         # Pydantic validation schemas
│   ├── tests/           # Pytest unit and integration test suite
│   └── utils/           # Embeddings, PDF parsers, LLM wrappers, and string helpers
├── frontend/
│   ├── src/
│   │   ├── api/         # Axios network service abstractions
│   │   ├── components/  # Modular layout, auth, upload, and tab components
│   │   ├── pages/       # Login, dashboard, and analysis layout hubs
│   │   ├── store/       # Zustand authentication stores
│   │   └── types.ts     # TypeScript interface declarations
│   └── public/          # Static assets and system icons
└── README.md            # System documentation
```

* **backend/**: Manages API execution, state database persistence, and AI/ML model evaluations.
* **frontend/**: Provides a responsive dashboard, upload wizards, and comparative bullet diff displays.

## Installation

### Clone Repository
```bash
git clone https://github.com/your-repo/MirrorCueAI.git
cd MirrorCueAI
```

### Database Setup
1. Create a PostgreSQL database instance locally or on Supabase.
2. Ensure you have the connection string ready.

### Backend Setup
1. Move to the backend folder:
   ```bash
   cd backend
   ```
2. Create and configure your `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
   *Fill in `DATABASE_URL`, `GEMINI_API_KEY`, and `JWT_SECRET`.*
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run migrations:
   ```bash
   alembic upgrade head
   ```
5. Run the backend server:
   ```bash
   python -m uvicorn backend.main:app --reload --port 8000
   ```

### Frontend Setup
1. Move to the frontend folder:
   ```bash
   cd ../frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
   *Note: Ensure to use `npm install` for frontend dependencies.*
3. Start the development server:
   ```bash
   npm run dev
   ```

## Application Workflow

```text
User 
  ↓ (Uploads Resume & JD)
Frontend
  ↓ (REST API Request)
Backend (FastAPI)
  ↓ (Triggers Async Agent Pipeline)
Database / AI (LangGraph State Machine)
  ↓ (Stores & Evaluates Model Results)
Response (Scoring & Interactive Q&A)
```

1. **Upload**: The user uploads their resume PDF and inputs the target JD in the frontend dashboard.
2. **Parsing & Vector Generation**: The backend extracts resume text, parses sections into structured JSON, and generates semantic sentence embeddings.
3. **Multi-Agent Evaluation**:
   - The **ATS Matcher** scores keyword coverage and semantic similarity.
   - The **Bias Mirror** extracts candidate features and runs the XGBoost model to flag potential bias vectors.
   - The **Q&A Agent** evaluates gaps and produces verification questions stored in the DB.
4. **Interactive Rewrite**: The frontend polls the API, renders results, collects Q&A answers, and submits them to trigger the **Rewrite Agent**, which outputs a polished, side-by-side bullet diff.

## API Overview

* **POST `/auth/register`**: Registers a new user.
* **POST `/auth/token`**: Authenticates user credentials and returns a JWT token.
* **POST `/resumes`**: Accepts PDF resume uploads, returning a unique file identifier.
* **POST `/analysis`**: Initiates the async LangGraph analysis pipeline for a resume against a job description.
* **GET `/analysis/{id}`**: Returns polling statuses, scores, and active questionnaires.
* **POST `/analysis/{id}/answers`**: Submits user answers to the resume gap questionnaire, triggering final bullet rewrites.

## AI Workflow

```text
[Input Resume & JD]
       ↓
[Parser Agent] ──(Structured JSON)──→ [ATS Matcher] ──(Keyword & Semantic Similarity)
       ↓                                    ↓
[Bias Mirror] ──(XGBoost Flags)────────→ [State Graph Accumulator]
       ↓                                    ↓
[Q&A Agent] ──(Gap Queries)────────────→ [Interactive User QA]
       ↓                                    ↓
[Rewrite Agent] ───────────────────────→ [Output Optimized Bullets]
```

1. **Input**: Unstructured resume PDF and target job description text.
2. **Processing**: The **Parser Agent** transforms the PDF text into structured JSON blocks.
3. **Analysis**: 
   - The **ATS Matcher** runs fuzzy matching algorithms across resume sections alongside vector similarity searches.
   - The **Bias Mirror** feeds computed candidate profiles to an XGBoost model.
   - The **Q&A Agent** identifies discrepancies and prompts the user.
4. **Output**: Consolidated scores, bias flags, and a final rewrite sheet displaying contextual resume updates.

## Testing

* **Testing Approach**: Employs automated unit and integration tests using Pytest, testing models, routers, and helper utilities.
* **Validation Process**: Verifies correctness of the parsing schemas, correctness of ATS scoring ratios, and performance limits of the embedding matches.
* **Quality Assurance Strategy**: Implements database rollback sessions in mock environments to ensure schema isolation during backend endpoints testing.

## Future Enhancements

### AI Mock Interview Assistant
Following resume matching and rewriting, users can access an AI-powered mock interview simulator. The system will:
* Generate company-specific and resume project-specific interview prompts.
* Simulate dynamic HR and technical rounds across approximately 6 interactive questions.
* Evaluate candidate inputs using LLM scoring to provide feedback on strengths and detailed recommendations.

## Conclusion

MirrorCue AI transforms resume analysis from a simple screening exercise into a comprehensive candidate intelligence workflow. Through its six-agent architecture, the platform evaluates job alignment, highlights potential bias signals, validates candidate context, and generates meaningful resume improvements backed by verified information. By bringing together ATS insights, recruiter-level feedback, and context-aware rewriting, MirrorCue AI helps students increase visibility, strengthen applications, and make better career opportunities more accessible.

