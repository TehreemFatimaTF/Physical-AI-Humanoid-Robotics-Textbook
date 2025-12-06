---
id: 1
title: Analyze Command Failed No Artifacts
stage: general
date: 2025-12-05
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: unknown
command: /sp.analyze
labels: [analysis, failure, no-artifacts]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - F:\hackathon\website\history\prompts\general\1-analyze-command-failed-no-artifacts.general.prompt.md
tests:

---

## Prompt

/sp.analyze

## Response snapshot

The feature directory and its associated specification, plan, and tasks files are missing. Please run `/sp.specify` first to create the necessary feature structure.

## Outcome

- ✅ Impact: Command execution failed due to missing prerequisite files.
- 🧪 Tests: N/A
- 📁 Files: PHR created.
- 🔁 Next prompts: User needs to run `/sp.specify`.
- 🧠 Reflection: The analysis command requires `spec.md`, `plan.md`, and `tasks.md` to exist. The error handling correctly identified the missing prerequisites.

## Evaluation notes (flywheel)

- Failure modes observed: Missing prerequisite files.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
