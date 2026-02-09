# Design Changelog

## 2026-02-06
- Centralized the GitHub-inspired theme in `dashboard/theme.css` and removed inline CSS from `dashboard/app.py`.
- Aligned palette to GitHub dark colors: canvas `#0d1117`, surfaces `#161b22`, borders `#30363d`, text `#c9d1d9`, accents `#58a6ff`.
- Standardized typography to system-ui, spacing to an 8px scale, and transitions to ~180ms.
- Restyled buttons, inputs, selects, tables, cards, badges, alerts, and expanders to match GitHub patterns.
- Added consistent connection/status pills, sidebar navigation styling, and scrollbars.
- Added `.streamlit/config.toml` to enforce dark theme defaults and improve consistency across environments.
- Replaced emoji-based labels with a minimal SVG icon system and standardized section headers.
