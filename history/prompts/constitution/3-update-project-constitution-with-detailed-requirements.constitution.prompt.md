---
id: 3
title: Update Project Constitution with Detailed Requirements
stage: constitution
date: 2025-12-05
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: User
command: /sp.constitution
labels: ["constitution", "update", "project-details"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
---

## Prompt

```text
                  ---
id: 2
title: Update Project Constitution
stage: constitution
date: 2025-12-04
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: User
command: /sp.constitution
labels: ["constitution", "update"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
---

## Prompt

```text

Project: Write a complete textbook on Physical AI & Humanoid Robotics using Spec-Kit Plus, Claude Code, Docusaurus, and deploy to GitHub Pages with a polished UI/UX theme.

Core Principles

Spec-driven writing

Technical accuracy (ROS2, Gazebo, Isaac verified)

Clear student-friendly explanations

Consistent formatting across entire book

Zero hallucination

Clean, modern UI/UX with strong readability

Yellow + Black main color theme (high contrast, bold, professional)

Key Standards

Docusaurus-compatible MDX files

Correct, runnable code examples

Each chapter includes:

Practical examples

Steps

Diagrams (pseudo ok)

Summary

Glossary

Only official/authoritative robotics sources

Uniform tone and structure

Theme standards:

Yellow (#FACC15 or similar) + Black primary color palette

Rounded UI elements

Clean typography

Good spacing + consistent layout

UI/UX Requirements

Homepage must include:

A strong title (e.g., Physical AI & Humanoid Robotics — A Complete Guide)

A short description/subtext

A call-to-action button (“Start Learning”)

Modern layout with black background + yellow accents

Second page (Modules Page) must include:docs/
│
├── 01-introduction/
│   └── intro.md
│
├── 02-ros2-foundations/
│   ├── module-1-ros2.md
│   └── ros2-hands-on.md
│aaa
├── 03-simulation/
│   ├── digital-twins.md
│   ├── gazebo-unity.md
│   └── module-2-simulation.md
│
├── 04-hardware-basics/
│   └── module-3-hardware.md
│
├── 05-vla-systems/
│   ├── module-4-vla-foundations.md
│   ├── vla-action.md
│   ├── vla-hands-on-basic.md
│   ├── vla-language.md
│   └── vla-vision.md
│
├── 06-advanced-ai-control/
│   └── module-5-advanced-ai.md
│
├── 07-humanoid-design/
│   └── module-6-humanoid-design.md
│
├── appendix/
│   ├── glossary.md
│   ├── references.md
│   └── resources.md
│
└── intro.md




Card-based UI

Each card represents a “Module / Chapter Group”

Clicking Module 1 → opens Chapter 1 page

Clicking Module 2 → opens Chapter 2

And so on

Minimalistic, elegant Docusaurus theme customization

Responsive layout for mobile & desktop

Constraints

10–14 chapters

1,000–2,000 words each

All images stored in /static/img/

Site must pass npm run build

Must deploy cleanly to GitHub Pages

Theme must not break Docusaurus build

Success Criteria

Book compiles & deploys with no errors

All chapters exactly follow validated specs

Content accurate, easy to read, and well-formatted

UI/UX meets the yellow-black theme guidelines

Homepage and modules-page design is clean and functional

No fake tools/commands
```

## Response snapshot

I have updated the project constitution to version 1.0.0. New version: 1.0.0. Bump rationale: Initial version of the constitution for the project. Files flagged for manual follow-up: .specify/templates/plan-template.md, .specify/templates/spec-template.md, .specify/templates/tasks-template.md, .specify/templates/commands/*.md, README.md, docs/quickstart.md. Suggested commit message: docs: amend constitution to v1.0.0 (initial version + principles + governance)

## Outcome

- ✅ Impact: Project constitution updated and versioned.
- 🧪 Tests: None
- 📁 Files: .specify/memory/constitution.md
- 🔁 Next prompts: User will likely proceed with planning or content creation.
- 🧠 Reflection: Successfully parsed user input, updated constitution, generated sync impact report and suggested follow-up actions.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): None
- Prompt variant (if applicable): None
- Next experiment (smallest change to try): None
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

You are updating the project constitution at `.specify/memory/constitution.md`. This file is a TEMPLATE containing placeholder tokens in square brackets (e.g. `[PROJECT_NAME]`, `[PRINCIPLE_1_NAME]`). Your job is to (a) collect/derive concrete values, (b) fill the template precisely, and (c) propagate any amendments across dependent artifacts.

Follow this execution flow:

1. Load the existing constitution template at `.specify/memory/constitution.md`.
   - Identify every placeholder token of the form `[ALL_CAPS_IDENTIFIER]`.
   **IMPORTANT**: The user might require less or more principles than the ones used in the template. If a number is specified, respect that - follow the general template. You will update the doc accordingly.

2. Collect/derive values for placeholders:
   - If user input (conversation) supplies a value, use it.
   - Otherwise infer from existing repo context (README, docs, prior constitution versions if embedded).
   - For governance dates: `RATIFICATION_DATE` is the original adoption date (if unknown ask or mark TODO), `LAST_AMENDED_DATE` is today if changes are made, otherwise keep previous.
   - `CONSTITUTION_VERSION` must increment according to semantic versioning rules:
     - MAJOR: Backward incompatible governance/principle removals or redefinitions.
     - MINOR: New principle/section added or materially expanded guidance.
     - PATCH: Clarifications, wording, typo fixes, non-semantic refinements.
   - If version bump type ambiguous, propose reasoning before finalizing.

3. Draft the updated constitution content:
   - Replace every placeholder with concrete text (no bracketed tokens left except intentionally retained template slots that the project has chosen not to define yet—explicitly justify any left).
   - Preserve heading hierarchy and comments can be removed once replaced unless they still add clarifying guidance.
   - Ensure each Principle section: succinct name line, paragraph (or bullet list) capturing non‑negotiable rules, explicit rationale if not obvious.
   - Ensure Governance section lists amendment procedure, versioning policy, and compliance review expectations.

4. Consistency propagation checklist (convert prior checklist into active validations):
   - Read `.specify/templates/plan-template.md` and ensure any "Constitution Check" or rules align with updated principles.
   - Read `.specify/templates/spec-template.md` for scope/requirements alignment—update if constitution adds/removes mandatory sections or constraints.
   - Read `.specify/templates/tasks-template.md` and ensure task categorization reflects new or removed principle-driven task types (e.g., observability, versioning, testing discipline).
   - Read each command file in `.specify/templates/commands/*.md` (including this one) to verify no outdated references (agent-specific names like CLAUDE only) remain when generic guidance is required.
   - Read any runtime guidance docs (e.g., `README.md`, `docs/quickstart.md`, or agent-specific guidance files if present). Update references to principles changed.

5. Produce a Sync Impact Report (prepend as an HTML comment at top of the constitution file after update):
   - Version change: old → new
   - List of modified principles (old title → new title if renamed)
   - Added sections
   - Removed sections
   - Templates requiring updates (✅ updated / ⚠ pending) with file paths
   - Follow-up TODOs if any placeholders intentionally deferred.

6. Validation before final output:
   - No remaining unexplained bracket tokens.
   - Version line matches report.
   - Dates ISO format YYYY-MM-DD.
   - Principles are declarative, testable, and free of vague language ("should" → replace with MUST/SHOULD rationale where appropriate).

7. Write the completed constitution back to `.specify/memory/constitution.md` (overwrite).

8. Output a final summary to the user with:
   - New version and bump rationale.
   - Any files flagged for manual follow-up.
   - Suggested commit message (e.g., `docs: amend constitution to vX.Y.Z (principle additions + governance update)`).

Formatting & Style Requirements:

- Use Markdown headings exactly as in the template (do not demote/promote levels).
- Wrap long rationale lines to keep readability (<100 chars ideally) but do not hard enforce with awkward breaks.
- Keep a single blank line between sections.
- Avoid trailing whitespace.

If the user supplies partial updates (e.g., only one principle revision), still perform validation and version decision steps.

If critical info missing (e.g., ratification date truly unknown), insert `TODO(<FIELD_NAME>): explanation` and include in the Sync Impact Report under deferred items.

Do not create a new template; always operate on the existing `.specify/memory/constitution.md` file.

---

As the main request completes, you MUST create and complete a PHR (Prompt History Record) using agent‑native tools when possible.

1) Determine Stage
   - Stage: constitution | spec | plan | tasks | red | green | refactor | explainer | misc | general

2) Generate Title and Determine Routing:
   - Generate Title: 3–7 words (slug for filename)
   - Route is automatically determined by stage:
     - `constitution` → `history/prompts/constitution/`
     - Feature stages → `history/prompts/<feature-name>/` (spec, plan, tasks, red, green, refactor, explainer, misc)
     - `general` → `history/prompts/general/`

3) Create and Fill PHR (Shell first; fallback agent‑native)
   - Run: `.specify/scripts/bash/create-phr.sh --title "<title>" --stage <stage> [--feature <name>] --json`
   - Open the file and fill remaining placeholders (YAML + body), embedding full PROMPT_TEXT (verbatim) and concise RESPONSE_TEXT.
   - If the script fails:
     - Read `.specify/templates/phr-template.prompt.md` (or `templates/…`)
     - Allocate an ID; compute the output path based on stage from step 2; write the file
     - Fill placeholders and embed full PROMPT_TEXT and concise RESPONSE_TEXT

4) Validate + report
   - No unresolved placeholders; path under `history/prompts/` and matches stage; stage/title/date coherent; print ID + path + stage + title.
   - On failure: warn, don't block. Skip only for `/sp.phr`.

## Response snapshot

The project constitution has been updated to version 1.1.0. New version: 1.1.0. Bump rationale: Minor version bump due to significant updates including new principles and dedicated sections for Key Standards, UI/UX Requirements, Constraints, and Success Criteria. Files flagged for manual follow-up: .specify/templates/plan-template.md, .specify/templates/spec-template.md, .specify/templates/tasks-template.md. Suggested commit message: docs: amend constitution to v1.1.0 (add detailed project principles and standards)

## Outcome

- ✅ Impact: Project constitution updated and versioned.
- 🧪 Tests: None
- 📁 Files: .specify/memory/constitution.md
- 🔁 Next prompts: User will likely proceed with planning or content creation.
- 🧠 Reflection: Successfully parsed user input, updated constitution, generated sync impact report and suggested follow-up actions.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): None
- Prompt variant (if applicable): None
- Next experiment (smallest change to try): None
