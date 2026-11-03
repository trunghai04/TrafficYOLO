# Project Guidelines - TrafficYOLO

This project follows strict engineering and visual design standards for Google Antigravity.

## UI/UX Engineering Guidelines
All user-facing interface modifications, new views, and component work must follow:
- Rules: [.agents/rules/ui-engineering.md](.agents/rules/ui-engineering.md)
- Global Skills Toolkit: `impeccable`, `frontend-design`, `ui-ux-pro-max`, `flutter-ai-ui-skill`, `flutter-visual-qa`
- Anti-AI-Slop standards: No generic cookie-cutter cards, arbitrary gradients, or emoji icons. Focus on strong typography, intentional whitespace, semantic color tokens, and rigorous visual QA.

## Technical Architecture
- **Core Domain:** Traffic video analytics, vehicle tracking (ByteTrack), license plate detection, and red-light violation evidence extraction.
- **Frontend / Application Interface:** Streamlit UI in `app.py`.
- **Quality & Testing:** Run pytest via `pytest tests/` before committing changes.
