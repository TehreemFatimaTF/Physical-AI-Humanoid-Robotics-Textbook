<!--
Sync Impact Report:
Version change: 1.0.0 -> 1.1.0
Modified principles:
  - I. Modularity -> I. Spec-Driven Development
  - II. User-Centric Design -> II. Technical Accuracy
  - III. Performance -> III. Clarity & Readability
  - IV. Security -> IV. Content Integrity
  - V. Maintainability -> V. UI/UX Excellence
  - None -> VI. Maintainability & Deployability
Added sections:
  - Key Standards
  - UI/UX Requirements
  - Constraints
  - Success Criteria
Removed sections: None
Templates requiring updates:
  - .specify/templates/plan-template.md: ⚠ pending (Constitution Check section might need alignment)
  - .specify/templates/spec-template.md: ⚠ pending
  - .specify/templates/tasks-template.md: ⚠ pending
Follow-up TODOs: None
-->
# Physical AI & Humanoid Robotics Textbook Constitution

## Core Principles

### I. Spec-Driven Development
All content and features must strictly adhere to validated specifications.

### II. Technical Accuracy
All technical information, especially regarding ROS2, Gazebo, and Isaac, must be verified against official and authoritative robotics sources.

### III. Clarity & Readability
Explanations must be student-friendly, clear, concise, and consistently formatted across the entire book to ensure strong readability.

### IV. Content Integrity
Maintain zero hallucination; all content must be factual and verifiable.

### V. UI/UX Excellence
Ensure a clean, modern, and responsive user interface with a high-contrast yellow and black color scheme, rounded UI elements, and clean typography.

### VI. Maintainability & Deployability
The project must be Docusaurus-compatible, pass `npm run build`, deploy cleanly to GitHub Pages, and have correct, runnable code examples that do not break the Docusaurus build.

## Key Standards

- Docusaurus-compatible MDX files.
- Correct, runnable code examples.
- Each chapter must include: Practical examples, steps, diagrams (pseudo ok), summary, and a glossary.
- Only official/authoritative robotics sources are to be used.
- Uniform tone and structure throughout the book.
- Theme standards: Yellow (#FACC15 or similar) and Black primary color palette, rounded UI elements, clean typography, good spacing, and consistent layout.

## UI/UX Requirements

- **Homepage**: Must include a strong title (e.g., "Physical AI & Humanoid Robotics — A Complete Guide"), a short description/subtext, a call-to-action button (“Start Learning”), and a modern layout with a black background and yellow accents.
- **Modules Page**: Must feature a card-based UI where each card represents a “Module / Chapter Group,” and clicking a module opens its corresponding chapter page.
- Minimalistic, elegant Docusaurus theme customization.
- Responsive layout for mobile and desktop.

## Constraints

- 10–14 chapters, each between 1,000–2,000 words.
- All images must be stored in `/static/img/`.
- The site must pass `npm run build`.
- Must deploy cleanly to GitHub Pages.
- The theme must not break the Docusaurus build.

## Success Criteria

- The book compiles and deploys with no errors.
- All chapters exactly follow validated specifications.
- Content is accurate, easy to read, and well-formatted.
- The UI/UX meets the yellow-black theme guidelines.
- Homepage and modules-page design is clean and functional.
- No fake tools or commands are present.

## Governance

This constitution supersedes all other practices. Amendments require documentation, approval, and a migration plan. All PRs/reviews must verify compliance. Use `CLAUDE.md` for runtime development guidance.

**Version**: 1.1.0 | **Ratified**: 2025-12-05 | **Last Amended**: 2025-12-05
