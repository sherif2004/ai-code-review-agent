# AI Code Review Agent by sherif

An\\ automated AI-powered code review system that analyzes GitHub Pull Requests,
generates structured feedback using LLMs, and posts review comments directly on PRs.

Built with FastAPI and GitHub Webhooks.

## Architecture

GitHub PR
   ↓
Webhook (FastAPI)
   ↓
Diff Extraction (GitHub API)
   ↓
LLM Review Engine
   ↓
PR Comment Posted Back to GitHub

## Features

- Detects Pull Request open/update events
- Extracts PR diffs via GitHub API
- Sends code changes to LLM for analysis
- Generates structured JSON feedback
- Posts automated review comments on PR

- ## Tech Stack

- Python
- FastAPI
- GitHub Webhooks
- GitHub REST API
- LLM API (OpenAI / OpenRouter / etc.)
- Railway / Render (deployment)

- ## Local Setup

1. Clone repository
2. Create virtual environment
3. Install dependencies:

pip install -r requirements.txt

4. Add environment variables:

GITHUB_TOKEN=...
LLM_API_KEY=...

5. Run server:

uvicorn main:app --reload

## GitHub Webhook Setup

1. Go to repository → Settings → Webhooks
2. Add webhook
3. Payload URL:

https://your-deployed-app/webhook

4. Select "Pull Requests" event

5. {
  "issues": [
    {
      "file": "main.py",
      "line": 42,
      "message": "Function lacks error handling."
    }
  ],
  "suggestions": [
    "Add try/except block around API call."
  ]
}

## Roadmap

- [x] Webhook integration
- [x] Diff extraction
- [x] LLM review
- [ ] Inline code comments
- [ ] Auto-refactoring suggestions
- [ ] Multi-agent review system

## Why This Project

This project demonstrates:

- Event-driven backend design
- GitHub API integration
- LLM reasoning over code diffs
- End-to-end automation
- Real-world DevOps workflow integration

  
