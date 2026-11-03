# UI/UX Engineering Toolkit for Google Antigravity
**Architecture, Evaluation, Installation, and Verification Report**

- **Date:** November 2, 2026
- **Specialist Roles:** Senior AI Agent Infrastructure Engineer, UI/UX Design Engineer, Frontend Architecture Specialist
- **Environment:** Windows 11, PowerShell, Node.js v22.18.0, Python 3.13.5, Git 2.50.1

---

## Executive Summary

To eliminate generic "AI-slop" (cookie-cutter rounded cards, arbitrary gradients, emoji-heavy layouts, missing interaction states, and untracked visual defects), a unified **UI/UX Engineering Toolkit** has been researched, installed, configured, and verified for Google Antigravity.

The toolkit provides:
1. **Design Intelligence & Anti-AI-Slop Directives:** Authoritative principles from Anthropic (`frontend-design`), Paul Bakaus (`impeccable`), and NextLevelBuilder (`ui-ux-pro-max`).
2. **Platform & Framework Specializations:** Deep design tokens, palettes, and typography for React/Next.js (`shadcn/ui`, `magicui`, motion primitives) and Flutter (`flutter-ai-ui-skill`, `flutter-visual-qa`).
3. **Automated Design Verification:** Native CLI tools and scripts (`impeccable`, `ui-ux-pro-max` search engine, `taste-skill` DOM inspection, Flutter visual QA matrix).
4. **Universal Antigravity Discovery:** Direct dual-path discovery via `~/.gemini/config/skills/`, explicit declaration via `~/.gemini/config/skills.json`, and ecosystem synchronization with `~/.agents/skills/`.

---

## 1. Environment Analysis

Prior to executing any modifications, the local machine and agent runtime environment were inspected:

| Dimension | Detected Environment | Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Pro (64-bit) | Local time: UTC+7 |
| **Shell & Execution** | PowerShell 5.1 / Windows Command Shell | Full execution privileges enabled |
| **Node.js Runtime** | Node.js v22.18.0 / npm 10.9.3 | Supported for Node scripts and CLI utilities |
| **Python Runtime** | Python 3.13.5 | Supported for data query scripts (`ui-ux-pro-max`) |
| **Version Control** | Git 2.50.1.windows.1 | Clean depth-1 clone capability verified |
| **Antigravity Config Root** | `C:\Users\ADMIN\.gemini\config\` | Native Antigravity machine-local configuration root |
| **Antigravity App Data** | `C:\Users\ADMIN\.gemini\antigravity\` | Contains internal state, builtin skills, and runtime bins |
| **Initial Global Skills** | `C:\Users\ADMIN\.agents\skills\` | Contained `find-skills` and `frontend-ui-engineering` |
| **Initial Project Root** | `C:\Users\ADMIN\Desktop\TrafficYOLO` | Python Streamlit traffic analytics project. No prior `.agents` or `GEMINI.md`. |

---

## 2. Comprehensive Repository Evaluation

All 13 candidate repositories were researched, cloned or inspected via API, and evaluated against Antigravity compatibility, maintenance status, security considerations, and duplication:

| # | Repository | Status | Classification | Compatibility | Recommendation & Decision |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **1** | `nextlevelbuilder/ui-ux-pro-max-skill` | **ACTIVE** | Agent Skill + CLI | Full (Antigravity/Python) | **INSTALLED (Global).** Master dataset with 79 styles, 192 palettes, 74 font pairings, 119 UX guidelines, and 22 technology stacks. |
| **2** | `anthropics/skills` | **ACTIVE** | Skill Monorepo | Full (Standard SKILL.md) | **INSTALLED (Global: `frontend-design`).** Industry benchmark for distinctive, non-templated visual design lead instructions. |
| **3** | `leonxlnx/taste-skill` | **ACTIVE** | Agent Skill + Scripts | Full (Node.js 22+) | **INSTALLED & LINKED (`design-taste-frontend`).** Anti-slop landing page art direction and automated DOM/screenshot inspector. |
| **4** | `pbakaus/impeccable` | **ACTIVE** | Agent Skill + Native CLI | Full (Native `.gemini` support) | **INSTALLED (Global).** The design vocabulary for coding agents (`polish`, `bolder`, `critique`, `audit`, `document`, `craft-floor`). |
| **5** | `shadcn-ui/ui` | **ACTIVE** | Component System | Project-level library | **DOCUMENTED.** Not an agent skill. Primary recommended React/Next.js accessible component system via `npx shadcn@latest`. |
| **6** | `magicuidesign/magicui` | **ACTIVE** | Component Library | Project-level library | **DOCUMENTED.** Project-level animated UI components and visual effects for design engineers. |
| **7** | `aceternity/ui` | **NOT ON GITHUB (404)** | Component Registry | Web Registry | **DOCUMENTED WITH FIX.** Repository `aceternity/ui` does not exist; Aceternity is hosted at `ui.aceternity.com` by Manu Arora (`mannupaaji`). |
| **8** | `ibelick/motion-primitives` | **ACTIVE** | Component Library | Project-level library | **DOCUMENTED.** Framer Motion / Motion primitives kit for React applications. |
| **9** | `21st-dev/magic-mcp` | **ACTIVE** | MCP Server | Stdio / HTTP MCP | **DOCUMENTED & CONFIGURED SAFELY.** Formerly Magic MCP, now 21st MCP (`@21st-dev/magic` / `21st CLI`). Requires API key; documented for safe opt-in. |
| **10** | `SpeakQuery/flutter-ai-ui-skill` | **ACTIVE** | Agent Skill + Datasets | Full (Antigravity/Dart) | **INSTALLED (Global).** Comprehensive master Flutter UI/UX skill with CSV design tokens, Material 3, and Cupertino blueprints. |
| **11** | `TheSimpleApp/flutter-app-skills` | **ACTIVE** | Skill Collection | Full | **SKIPPED (Duplicate).** 4 narrow skills whose capabilities are completely covered by `flutter-ai-ui-skill` and `flutter-design-engineer`. |
| **12** | `musabekisakov-imj/flutter-design-engineer` | **ACTIVE** | Agent Skill Suite | Full | **INSTALLED (`flutter-visual-qa`).** High-value visual QA matrix, screenshot verification, and golden testing workflow. |
| **13** | `flutter/samples` | **ACTIVE** | Code Reference | Reference Project | **DOCUMENTED AS REFERENCE.** Official Flutter production sample apps. Kept as architectural benchmark rather than agent skill. |

---

## 3. Selected Skills and Architectural Rationale

The final selection avoids cognitive clutter and token exhaustion while providing end-to-end coverage across the design lifecycle:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   ANTIGRAVITY UI/UX ENGINEERING TOOLKIT                │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Art Direction & Taste:                                              │
│    - frontend-design (Anthropic): Distinctive aesthetics, anti-default  │
│    - design-taste-frontend (Taste Skill): Hero compositions & DOM check│
├────────────────────────────────────────────────────────────────────────┤
│ 2. Design Intelligence & Tokens:                                       │
│    - ui-ux-pro-max: 192 palettes, 74 font pairings, 22 stack tokens    │
│    - frontend-ui-engineering: Production component architecture (coloc) │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Polish, Critique & Heuristic Vocabulary:                           │
│    - impeccable: Standard commands (critique, audit, polish, bolder)    │
│    - impeccable craft-floor: Absolute quality floor before edits        │
├────────────────────────────────────────────────────────────────────────┤
│ 4. Mobile & Flutter Specialization:                                    │
│    - flutter-ai-ui-skill: Cupertino/Material 3 intelligence & CSVs     │
│    - flutter-visual-qa: Viewport matrix, golden tests & visual proof    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Directory Structure & Dual-Path Discovery

To guarantee that Google Antigravity (IDE, Desktop 2.0, CLI) and ecosystem tools (`npx skills`) discover all skills simultaneously without duplicating disk storage:

1. **Ecosystem Root (`~/.agents/skills/`):** Houses the installed skills with complete script, reference, and data assets.
2. **Native Antigravity Root (`~/.gemini/config/skills/`):** Windows directory junctions (`mklink /J`) point directly to the ecosystem root, granting Antigravity direct native directory traversal.
3. **Antigravity JSON Manifest (`~/.gemini/config/skills.json`):** Explicitly declares both paths so any session honors them across all discovery tiers.

```
C:\Users\ADMIN\
├── .agents\
│   └── skills\
│       ├── design-taste-frontend\        [SKILL.md, scripts/, library/, references/]
│       ├── find-skills\                  [SKILL.md]
│       ├── flutter-ai-ui-skill\          [SKILL.md, data/, examples/, templates/]
│       ├── flutter-visual-qa\            [SKILL.md, references/]
│       ├── frontend-design\              [SKILL.md]
│       ├── frontend-ui-engineering\      [SKILL.md]
│       ├── impeccable\                   [SKILL.md, reference/, scripts/]
│       └── ui-ux-pro-max\                [SKILL.md, data/, scripts/, references/]
│
├── .gemini\
│   └── config\
│       ├── skills.json                   [Explicit Antigravity skill path registry]
│       └── skills\
│           ├── design-taste-frontend     <=== Junction ===> ~/.agents/skills/design-taste-frontend
│           ├── find-skills               <=== Junction ===> ~/.agents/skills/find-skills
│           ├── flutter-ai-ui-skill       <=== Junction ===> ~/.agents/skills/flutter-ai-ui-skill
│           ├── flutter-visual-qa         <=== Junction ===> ~/.agents/skills/flutter-visual-qa
│           ├── frontend-design           <=== Junction ===> ~/.agents/skills/frontend-design
│           ├── frontend-ui-engineering   <=== Junction ===> ~/.agents/skills/frontend-ui-engineering
│           ├── impeccable                <=== Junction ===> ~/.agents/skills/impeccable
│           └── ui-ux-pro-max             <=== Junction ===> ~/.agents/skills/ui-ux-pro-max
│
└── Desktop\TrafficYOLO\
    ├── GEMINI.md                         [Project-level instructions pointing to rules]
    ├── UI_ENGINEERING_TOOLKIT_REPORT.md  [This authoritative report]
    └── .agents\
        └── rules\
            └── ui-engineering.md         [Strict 6-Stage UI Workflow & Anti-AI-Slop Rules]
```

---

## 5. Installation Commands Executed

The following non-destructive commands were executed to establish the toolkit:

```powershell
# 1. Install Skills via Ecosystem Package Manager
npx skills add anthropics/skills --skill frontend-design -g -y
npx skills add SpeakQuery/flutter-ai-ui-skill -g -y
npx skills add pbakaus/impeccable -g -y
npx skills add nextlevelbuilder/ui-ux-pro-max-skill --skill ui-ux-pro-max -g -y
npx skills add musabekisakov-imj/flutter-design-engineer --skill flutter-visual-qa -g -y
npx skills add leonxlnx/taste-skill -g -y

# 2. Establish Antigravity Global Customization Root & Junctions
New-Item -ItemType Directory -Path "$env:USERPROFILE\.gemini\config\skills" -Force

Get-ChildItem -Path "$env:USERPROFILE\.agents\skills" -Directory | ForEach-Object {
    $target = $_.FullName
    $link = Join-Path "$env:USERPROFILE\.gemini\config\skills" $_.Name
    if (-not (Test-Path $link)) {
        cmd /c "mklink /J `"$link`" `"$target`""
    }
}

# 3. Create Antigravity Global Manifest (~/.gemini/config/skills.json)
# Configured with explicit path entries to ~/.gemini/config/skills and ~/.agents/skills
```

---

## 6. Dependencies & Runtime Requirements

| Skill / Tool | Dependencies | Minimum Version | Verified Status |
| :--- | :--- | :--- | :--- |
| **`impeccable`** | Self-contained launcher (`impeccable.cmd`), zero external deps | v4.8.0 | Verified (`4.8.0` returned) |
| **`ui-ux-pro-max`** | Python 3 standard library (`csv`, `json`, `argparse`) | Python 3.8+ | Verified (Python 3.13.5) |
| **`frontend-design`** | Pure Markdown instruction | Any LLM runtime | Verified (Valid YAML frontmatter) |
| **`flutter-ai-ui-skill`** | Local CSV data files | Any Flutter SDK | Verified (All 4 CSVs intact) |
| **`flutter-visual-qa`** | Pure Markdown instruction + Flutter test environment | Flutter SDK | Verified |
| **`design-taste-frontend`** | Node.js 22+, Chromium / Chrome for DOM inspector scripts | Node 22.18.0 | Verified |
| **`21st MCP`** (Optional) | Node.js / `npx -y @21st-dev/magic@latest` + API Key | Node 18+ | Documented below |

---

## 7. Antigravity Configuration Changes

### A. Global Skill Registry (`~/.gemini/config/skills.json`)
Created the declarative manifest in `C:\Users\ADMIN\.gemini\config\skills.json`:
```json
{
  "entries": [
    {
      "path": "~/.gemini/config/skills"
    },
    {
      "path": "~/.agents/skills"
    }
  ]
}
```

### B. Project Rules (`.agents/rules/ui-engineering.md`)
Created project-scoped rules establishing the **6-Stage UI Engineering Workflow** and **Anti-AI-Slop Rules** to automatically govern all future interface modifications.

### C. Project Guidance (`GEMINI.md`)
Created `c:\Users\ADMIN\Desktop\TrafficYOLO\GEMINI.md` to guide agents operating in this workspace.

### D. MCP Configuration (`mcp_config.json`)
The existing `mcp_config.json` was inspected and preserved intact without inserting dummy or unauthorized credentials. To enable the 21st MCP for UI component retrieval, the user can add:

```json
{
  "mcpServers": {
    "21st": {
      "command": "npx",
      "args": ["-y", "@21st-dev/magic@latest"],
      "env": {
        "API_KEY_21ST": "YOUR_API_KEY_FROM_21ST_DEV"
      }
    }
  }
}
```

---

## 8. Verification Results

Every installed skill and utility was systematically verified:

| Test Target | Verification Command / Procedure | Result | Output Evidence |
| :--- | :--- | :---: | :--- |
| **Global Discovery Registry** | `npx skills ls -g` | **PASS** | Listed 8 skills across Antigravity, Cline, Cursor, Copilot |
| **Antigravity Path Traversal** | Python validation of all 8 `SKILL.md` files in both directories | **PASS** | 100% valid YAML frontmatter, non-empty bodies, verified permissions |
| **`impeccable` CLI** | `impeccable.cmd --version` | **PASS** | `4.8.0` exited with code 0 |
| **`ui-ux-pro-max` Search** | `python search.py "dashboard" --domain style` | **PASS** | Returned 2 verified design specifications with tokens |
| **`ui-ux-pro-max` Color** | `python search.py "palette fintech" --domain color` | **PASS** | Returned semantic color palette with contrast notes |
| **Flutter Data Integrity** | Directory listing of `flutter-ai-ui-skill/data` | **PASS** | `flutter_colors.csv`, `flutter_typography.csv`, `flutter_components.csv`, `flutter_animations.csv` verified |
| **Project Rule Integrity** | File syntax & path resolution in `.agents/rules/ui-engineering.md` | **PASS** | Validated against Antigravity 24KB rule budget |

---

## 9. Unified UI Engineering Workflow

Every agent modifying or designing a UI must execute these 6 stages:

```
[Stage A: Project Analysis]
   │  Detect framework (React, Flutter, Streamlit), existing tokens & components.
   ▼
[Stage B: Art Direction]
   │  Query 'ui-ux-pro-max' / 'frontend-design' to select custom aesthetic (no AI slop).
   ▼
[Stage C: Design System]
   │  Define strict semantic tokens (colors, typography, spacing, surface contrast).
   ▼
[Stage D: Implementation]
   │  Colocate components; implement All states (Loading, Empty, Error, Active).
   ▼
[Stage E: Visual QA]
   │  Run real app; capture renders at 3 viewports (Mobile, Tablet, Desktop).
   ▼
[Stage F: Critique & Refinement]
      Run 'impeccable critique' & 'impeccable craft-floor' to reach human-grade polish.
```

### Anti-AI-Slop Heuristics
1. **Never default to cream backgrounds with serif display and terracotta accents** (Anthropic Claude default tell).
2. **Never default to near-black backgrounds with neon acid-green or vermilion accents**.
3. **No generic SaaS card kit:** Do not chop all content into identical cards with identical `border-radius: 16px` and soft grey drop shadows.
4. **No tracked-out ALL CAPS labels:** Avoid `WORD — fragment` or `01 / 02 / 03` unless it is a literal ordered sequence.
5. **No emoji substituting for UI icons:** Always use Lucide, Material Icons, or Cupertino vector icons.

---

## 10. Instructions for Future Projects

To apply this toolkit to any new project:

1. **Automatic Global Discovery:**
   Because the toolkit is installed globally in `~/.gemini/config/skills/` and `~/.agents/skills/`, all skills (`impeccable`, `frontend-design`, `ui-ux-pro-max`, `flutter-ai-ui-skill`, `flutter-visual-qa`) are instantly accessible to Antigravity in **any project on this machine**.
2. **Adding Project Rules:**
   Copy `.agents/rules/ui-engineering.md` into `<new-project>/.agents/rules/ui-engineering.md`.
3. **Querying Design Intelligence On-Demand:**
   - Query design styles: `python ~/.gemini/config/skills/ui-ux-pro-max/scripts/search.py "<query>" --domain style`
   - Query color palettes: `python ~/.gemini/config/skills/ui-ux-pro-max/scripts/search.py "<product>" --domain color`
   - Query typography: `python ~/.gemini/config/skills/ui-ux-pro-max/scripts/search.py "<mood>" --domain typography`
   - Run UI audit: `~/.gemini/config/skills/impeccable/scripts/impeccable.cmd critique --target <source_file>`
4. **Project Component Libraries:**
   - For React/Next.js: run `npx shadcn@latest init` within the project.
   - For Flutter: run `flutter pub add google_fonts` and import `flutter_colors.csv` themes.

---

## 11. Known Limitations & Safe Operations

1. **GitHub API Rate Limits:** Unauthenticated GitHub API calls may hit rate limits (60 req/hr). Git shallow clones (`git clone --depth 1`) or `npx skills add` are preferred and immune to API header rate limits.
2. **Aceternity UI:** `aceternity/ui` does not exist as a GitHub repository. Components should be browsed at `ui.aceternity.com` and installed via copy-paste or their registry CLI.
3. **MCP Credentials:** The 21st MCP requires a valid API key from `21st.dev/mcp` for persistent private component access. It was intentionally documented rather than configured with non-functional credentials.
4. **Headless Browser Capture on Windows:** `taste-skill` inspection scripts (`inspect.mjs`) require a locally installed Chrome/Chromium executable. If running in environments without Chrome, use Antigravity's built-in screenshot tools or Flutter golden tests.
