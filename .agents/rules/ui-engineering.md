# Professional UI/UX Engineering Rules & Workflow

This project enforces production-grade UI/UX engineering standards. When creating, modifying, or auditing any user interface (Web, Streamlit, React, Next.js, Flutter, or Mobile), every agent must strictly follow this workflow and adhere to the Anti-AI-Slop design heuristics.

---

## 1. Six-Stage UI Engineering Workflow

### Stage A: Project Analysis
- **Inspect Existing Architecture:** Detect UI framework (e.g., Streamlit, React, Next.js, Flutter) and existing styling systems before writing code.
- **Analyze Existing Tokens:** Check color palettes, font stacks, spacing scales, and layout conventions. Never introduce conflicting design patterns.
- **Identify Reusable Components:** Search for existing UI elements and utilities before creating new ones.
- **Preserve Business Logic:** Never alter domain logic, backend APIs, or data flows when performing visual upgrades.

### Stage B: Art Direction
- **Ground Design in Subject Matter:** Visual direction must be derived directly from the application's domain, target audience, and primary task—not from generic SaaS templates.
- **Define Design Parameters Upfront:**
  - Visual hierarchy and layout mode (Persuade, Operate, Read, or Experience).
  - Typography system (max 2 distinct font families with deliberate hierarchy).
  - Semantic color palette (4–6 core colors with explicit contrast ratios ≥ 4.5:1).
  - Spacing rhythm (strict 4px or 8px grid).
- **Reject AI Tells:** Eliminate arbitrary gradient borders, generic icon badges, and cookie-cutter layouts.

### Stage C: Design System & Tokens
- **Typography Tokens:** Establish consistent `display`, `heading`, `body`, and `caption` scales with proportional line heights.
- **Color Tokens:** Assign explicit semantic roles (`background`, `surface`, `primary`, `secondary`, `accent`, `border`, `destructive`, `muted`).
- **Surface Elevation:** Use subtle contrast or a single cohesive border strategy rather than arbitrary deep shadows.
- **Icons & Affordances:** Use meaningful SVG / framework icons with semantic labels. Do not use emoji as interactive icons or structural decor.

### Stage D: Implementation
- **Component Colocation & Composition:** Favor composition over configuration. Colocate component-related types, styles, and helpers.
- **State Completeness:** Every interface element must implement all required states: Default, Hover/Focus, Active, Loading, Empty, and Error.
- **Accessibility by Default:** Keyboard navigation, visible focus indicators, screen-reader text (`aria-label` / `Semantics`), and minimum touch targets (44×44px / 48×48dp).

### Stage E: Visual QA
- **Rendered Output is Truth:** Code inspection alone does not guarantee visual excellence.
- **Run the Real Interface:** Launch the local application or dev server and inspect actual renders.
- **Multi-Viewport & State Check:** Test at minimum three viewport widths (Mobile 375px, Tablet 768px, Desktop 1280px+) and check both empty and loaded data states.
- **Inspect Alignment & Spacing:** Check for text truncation, overflow, clipping, inconsistent gaps, and contrast failures.

### Stage F: Critique & Refinement
- **Self-Critique:** Review the rendered UI against human design benchmarks (e.g., Apple HIG, Material 3, curated design standards).
- **Heuristic Quality Floor:** Run `impeccable critique`, `impeccable polish`, or `ui-ux-pro-max` checks.
- **Batch Adjustments:** Fix identified issues in a focused refinement pass and re-verify visually.

---

## 2. Anti-AI-Slop Rules

The following tropes are strictly prohibited unless explicitly requested by the user:

| Prohibited Pattern | Why It Fails | Approved Alternative |
| :--- | :--- | :--- |
| **Endless rounded cards** | Chops content into repetitive boxes with equal weight. | Clear typography hierarchy, subtle dividers, and structured whitespace. |
| **Arbitrary multi-color gradients** | Pure visual noise without functional meaning. | Solid semantic brand colors or subtle tonal shifts. |
| **Heavy frosted glass (Glassmorphism)** | Destroys readability and legibility on diverse backgrounds. | Clean solid surfaces with crisp, accessible 1px borders. |
| **Overbearing soft drop shadows** | Makes flat screens look muddy and amateurish. | Precise elevation hierarchy using subtle 1px border or tonal elevation. |
| **Tracked-out ALL CAPS labels** | AI cliché used as decorative filler (`FEATURES`, `OVERVIEW`). | Natural sentence case or meaningful semantic subheadings. |
| **Emoji as interactive icons** | Inconsistent across platforms, lacks accessibility, looks juvenile. | Proper vector icons (Lucide, Material Icons, CupertinoIcons, Heroicons). |
| **Meaningless stat counters with arbitrary stats** | Generic vanity numbers disconnected from user value. | Real, verifiable data or contextual workflow metrics. |
| **Gratuitous entry animations** | Staggers and slides that delay users completing tasks. | Purposeful micro-interactions with duration under 250ms and reduced-motion support. |

---

## 3. Framework-Specific Standards

### React & Next.js
- **Component Architecture:** Leverage `shadcn/ui` and Radix UI primitives with Tailwind CSS.
- **Motion:** Use motion primitives (`framer-motion` or Motion) with purposeful, non-distracting physics.
- **Design Tokens:** Centralize tokens in `tailwind.config.ts` or CSS variables (`globals.css`). Never sprinkle hardcoded arbitrary hex values across components.

### Flutter
- **Design Language:** Strictly implement Material 3 (`useMaterial3: true`) or Cupertino (`CupertinoThemeData`). For iOS targets, prioritize native gesture feedback and Cupertino slivers.
- **Adaptive Layouts:** Use `LayoutBuilder`, `MediaQuery`, and adaptive scaffolds. Never fix screen dimensions in logical pixels.
- **Typography & Theme:** Structure typography through `Theme.of(context).textTheme`. Load design system palettes via `ColorScheme.fromSeed`.
- **Visual QA:** Execute golden tests or snapshot verification across compact and expanded constraints.

### Python UI (Streamlit / Gradio)
- **Layout Precision:** Use `st.columns`, containers, and custom CSS injections with semantic variables.
- **Visual Rhythm:** Avoid default cluttered vertical stacks. Group controls into sidebar or modular expandable sections with consistent padding and typography.
- **Status Communication:** Use semantic alerts (`st.success`, `st.warning`, `st.error`) with clear actionable guidance, not decorative text.
