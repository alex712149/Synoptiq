# SkyBlend Insights

SkillBlend — Master Prompt for Lovable  for  the backend zip file

Paste this whole block into Lovable as your starting prompt. This replaces the previous version — the backend was rebuilt with a different API contract (base path, field names, and two new capabilities worth designing around: an honest case-by-case replay that sometimes admits SkillBlend didn't win, and a dedicated weight-map endpoint across the full lead-time ladder).

Project context

Build the frontend for SkillBlend, an AI system built for Smart India Hackathon (PS 26081, NCMRWF / Ministry of Earth Sciences). SkillBlend takes forecasts from multiple weather models (GFS, IFS, AIFS), learns which model to trust for the current region/season/lead-time/weather-regime, and blends them into a single smarter forecast — with full transparency into why it trusted what it trusted, and full honesty about when it didn't win.

This is not a generic SaaS dashboard. This is a weather-intelligence command center — it should feel like something between a meteorological situation room, a sci-fi mission control, and a beautifully designed data-observatory. Think: the drama of watching a storm form on a live radar, crossed with the polish of a premium product launch site.

Do not lock in specific CSS values, exact hex codes, or fixed pixel measurements in this brief. I'm describing mood, motion, and hierarchy — you have full creative authority over the actual color system, typography choices, spacing scale, and implementation details. Surprise me with something sophisticated, not literal.

Overall creative direction

Mood: atmospheric, high-stakes, intelligent, alive. Not corporate-dashboard-blue-and-white. Think dark, immersive, glowing — like standing inside a storm-tracking control room at 2am watching a cyclone approach.

Reference feelings, not references to copy: the depth of a space-mission dashboard, the confident motion design of a premium product launch (Apple/Linear/Vercel-tier polish), the raw energy of watching live weather radar sweep across a screen.

Palette direction (not literal): dark base with something that feels like storm light — electric cyan/teal for "trust and clarity," a warning amber/red for "bust risk and disagreement," a deep violet-blue for the atmosphere itself. Let gradients feel like they're breathing, not static.

Typography: something with real character for headlines — a touch of technical/monospace energy for data readouts (this is a forecasting instrument, the numbers should feel instrumented, not decorative) paired with a clean, confident sans for body copy.

This should NOT feel like a template. No generic SaaS hero-with-stock-illustration. Every screen should feel purpose-built for weather intelligence.

A note on honesty as a design value: this system sometimes reports that it didn't beat a single model on a given case (see Replay below). Don't hide or downplay this in the UI — treat "honest reporting" as a badge of scientific credibility, not a bug to disguise. A small "credibility" visual language (a distinct but calm treatment for honest-miss cases, not shame-red) will actually make the win cases land harder.

The landing page — this is the showpiece

This needs to be the coolest, most sophisticated landing page you can build. Full creative freedom on execution, but it must include:

A 3D hero centerpiece. A rotating/interactive 3D Earth or atmospheric globe (react-three-fiber / drei), with visible storm systems, swirling cloud-like particle layers, or animated weather bands moving across its surface. It should respond subtly to mouse movement/scroll (parallax tilt, not just spin). This is the "wow" moment — the thing a judge remembers.

Scroll-driven storytelling. As the user scrolls past the hero, sections should reveal with scroll-triggered animation (GSAP ScrollTrigger or Framer Motion whileInView) — not simple fade-ins, but genuinely choreographed reveals: elements that feel like they're being tracked and assembled by radar, data points that "lock on," panels that slide in like HUD elements powering up.

Parallax depth layers. Background atmospheric elements (cloud textures, particle fields representing rain/wind, subtle grid/radar-sweep overlays) should move at different speeds than foreground content as the user scrolls, creating real depth — not a single flat parallax hero image.

Live-feeling data glimpses embedded in the landing page itself. Don't just describe the product in text — show a stylized, animated preview of the actual weight-blending happening: three model "signals" (GFS/IFS/AIFS) visually converging into one blended signal, with an animated confidence/trust meter ticking up. This should look like it's actually computing, even if it's a beautifully looped animation on the landing page.

Micro-interactions everywhere. Buttons that ripple/glow on hover with real physicality, cursor-aware hover states on cards, numbers that count up when they scroll into view, a subtle ambient animation that never fully stops (the page should feel "alive" even when idle — like a live instrument, not a static poster).

A confident, dramatic narrative flow: Hero (the big 3D moment + one-line pitch) → The Problem (visualize "no single model is always right" — maybe three model outputs visibly disagreeing) → The Solution (the convergence animation) → Live Capability Highlights (trust scoring, SHAP explainability, bust replay, weight maps) → Proof (the measured CSI improvement numbers, presented like a mission-readout stat block — including at least one honest "didn't win this one" case, framed as rigor not weakness) → CTA into the live dashboard.

A closing section that feels like a "systems ready" moment — transitioning the user from marketing site into the actual live app, ideally with a transition animation that feels like "entering" the dashboard (camera push-in, zoom-through-the-globe, whatever fits) rather than a flat page navigation.

App pages (post-landing)

1. Live Forecast Dashboard

Map-based interface (MapLibre/Mapbox) showing the 3 pilot regions — KWG (Kerala/Western Ghats), BOB (Bay of Bengal/East Coast), IGP (Indo-Gangetic Plains/NW India) — as selectable zones, plotted at their actual coordinates.

Region / variable (precipitation, temperature, wind_speed) / lead-time selectors, using the real lead-time ladder: 24, 48, 72, 96, 120, 144, 168 hours. Make these feel like instrument controls, not basic dropdowns.

Per-source breakdown panel — for each of GFS/IFS/AIFS, show forecast_value, weight, and historical_skill side by side. Some weights will legitimately be exactly 0.0 (a model fully excluded from the blend for this context) — design for that as a real, meaningful state (e.g. the bar visibly empties/greys rather than looking broken).

Trust breakdown — the response gives five separate components (historical_skill_component, disagreement_component, lead_time_component, regime_stability_component, data_quality_component) plus the final trust_score. Visualize this as a composed instrument — e.g. five small gauges/rings feeding into one larger trust dial — not a single flat number. This is a richer signal than before; give it real visual weight.

Forecast readout — show raw_blend_value, bias_corrected_value, and final_value as a visible pipeline (raw → corrected → final), so the bias-correction step is legible, not hidden.

Bust risk — bust_probability and bust_flag; when abstain: true, the UI should visibly shift into a "low confidence — human review" state rather than presenting a falsely confident number.

Explanation panel — render the explanation array (feature, contribution, direction, detail) as an explainable breakdown — a radar-style signal decomposition, animated in as bars.

Provenance footer — small, unobtrusive display of provenance (which real data source backs each model, which regime-detector/explanation backend is live) — a nice "this is real, not fake" trust signal for judges who read closely.

Regime panel — regime (the winning classification) plus the full regime_probs distribution, shown as a small probability spread, not just the single winning label.

2. Weight Map Explorer (new — build this as its own dedicated screen)

Powered by GET /weights/map?region=&variable=&season=&regime=, which returns weights across the entire lead-time ladder at once (24h through 168h) for a fixed context.

Visualize this as an animated multi-line or stacked-area chart showing how each model's trust/weight evolves as lead time increases — this is a genuinely new, interesting shape of data (e.g. IFS might dominate at 24h and stay dominant out to 168h, or another model might overtake it further out). Let the user change region/season/regime and watch the whole curve family re-draw with a satisfying morphing transition.

Also show the trust_score per lead-time point on the same view (secondary axis or a synced small-multiple) — trust visibly decaying with lead time is a real, tell-able story.

3. Counterfactual Bust Replay

Powered by GET /replay/events (list) and GET /replay/events/{event_id} (detail). Treat this as a cinematic "replay console" — this is your single most convincing feature, build it like a highlight reel.

The events list gives a label and a headline per case — use the headline text directly, it's already written as honest human-readable framing (e.g. "SkillBlend beat the naive average but not the single best model on this particular case" vs. "SkillBlend cut the error vs. both the naive average and the default single model"). Give these two headline "flavors" distinct but tonally calm visual treatments (a clear win vs. an honest miss) — don't force every case into a triumphant frame.

The detail view gives you everything needed for a proper side-by-side: raw_sources (three model forecasts), naive_average, single_model_choice, skillblend_blend, reference_value (the verified truth), and the three error values (naive_average_error, single_model_error, skillblend_error). Render this as a competing-bars-converging-on-the-truth-line visual — four bars (naive, single-best, SkillBlend, actual truth) racing toward the same target, animated in sequence.

The narrative field is a pre-written array of sentences walking through exactly this story step by step — reveal these one at a time, timed with the bar animations, like a live mission commentary track rather than a wall of text dumped at once.

4. Verification Scorecard

Powered by GET /skill/verification?region=&variable= — returns best_single_model, best_single_model_csi, skillblend_csi, relative_csi_improvement, meets_target, target_relative_csi_improvement per region/variable.

Note this is now honestly mixed across region/variable combinations — some clear wins (+18.5%, +33.7%, +36.1%), some near-target, some negative (wind_speed showed -4.3% and -16.7% in real testing). Design a clean way to show this full honest matrix — a grid/heatmap across region × variable, color-scaled by relative improvement, with the target line clearly marked, so the overall story ("wins decisively on rainfall, mixed on wind") reads at a glance rather than needing every judge to read every cell.

Highlight the headline wins prominently (rainfall improvements especially) while keeping the full matrix visible and honest — this is a feature, not something to bury.

5. Extreme Weather Guidance

Powered by GET /extreme/guidance?region=&lead_hours= — returns calibrated exceedance probability per variable (P(rain>50mm), P(temp>40°C), P(wind>17m/s)), each flagged calibrated: true/false.

Present as three threat-level instruments with escalating color intensity from calm → alert. Visibly distinguish calibrated vs. uncalibrated readings (e.g. a subtle badge/texture difference) — don't silently present both with equal confidence framing.

6. About / Architecture

A visual explainer of the pipeline (Forecast Sources → Normalization → Skill Engine → Regime Context → Adaptive Meta-Model → Bias Correction → Blended Output + Trust + Bust Risk) as an animated flow diagram that builds itself in as the user scrolls — echo the landing page's motion language here.

Data integration — real endpoints, real shapes

Base path: /api/v1

GET /regions                    → [{ code, name, lat, lon, emphasis, coastal }]

GET /sources                    → { GFS: {kind, provider, role}, IFS: {...}, AIFS: {...} }

GET /variables                  → [{ variable, unit, extreme_threshold }]

GET /lead-times                 → [24, 48, 72, 96, 120, 144, 168]

GET /health                     → readiness check

GET /forecast/blend?region=&variable=&lead_hours=

GET /weights/map?region=&variable=&season=&regime=      (season & regime are REQUIRED)

GET /skill/scorecard?...

GET /skill/verification?region=&variable=

GET /extreme/guidance?region=&lead_hours=

GET /replay/events

GET /replay/events/{event_id}

Region codes are KWG, BOB, IGP — not slugs. Variable values are precipitation, temperature, wind_speed.

Key response shape from GET /forecast/blend (design components around this exact structure):

json

{

  "region": "KWG", "region_name": "Kerala / Western Ghats",

  "variable": "precipitation", "unit": "mm/24h",

  "run_time": "...", "valid_time": "...", "lead_hours": 72,

  "season": "sw_monsoon", "regime": "active_monsoon",

  "regime_probs": { "active_monsoon": 0.2, "break_monsoon": 0.2, "western_disturbance": 0.2, "depression": 0.2, "normal": 0.2 },

  "sources": [

    { "model": "AIFS", "forecast_value": 14.512, "weight": 0.0, "historical_skill": 0.3395 },

    { "model": "GFS", "forecast_value": 0.0, "weight": 0.022, "historical_skill": 0.5186 },

    { "model": "IFS", "forecast_value": 1.281, "weight": 0.978, "historical_skill": 0.649 }

  ],

  "disagreement": 6.56,

  "raw_blend_value": 1.253, "bias_corrected_value": 5.847, "final_value": 5.847,

  "trust": {

    "historical_skill_component": 0.6461, "disagreement_component": 0.7376,

    "lead_time_component": 0.5714, "regime_stability_component": 0.0,

    "data_quality_component": 1.0, "trust_score": 0.5963

  },

  "bust_probability": 0.0, "bust_flag": false, "abstain": false,

  "explanation": [

    { "feature": "historical_skill_avg", "contribution": -0.0008, "direction": "increases_weight", "detail": "..." }

  ],

  "fallback_used": false,

  "provenance": { "AIFS": "ECMWF Open Data", "GFS": "NOAA NOMADS", "IFS": "ECMWF Open Data", "regime_detector": "...", "explanation_backend": "shap.TreeExplainer" }

}

GET /weights/map response shape:

json

{

  "region": "KWG", "variable": "precipitation", "regime": "active_monsoon", "season": "sw_monsoon",

  "points": [

    { "lead_hours": 24, "weights": { "GFS": 0.083, "IFS": 0.917, "AIFS": 0.0 }, "trust_score": 0.8036 },

    { "lead_hours": 48, "weights": { "GFS": 0.031, "IFS": 0.969, "AIFS": 0.0 }, "trust_score": 0.7693 }

  ]

}

GET /replay/events/{id} response shape:

json

{

  "event_id": "EVT-BOB-20260503-L120", "region": "BOB", "variable": "precipitation", "unit": "mm/24h",

  "valid_time": "...", "lead_hours": 120,

  "label": "Bay of Bengal / East Coast - 2026-05-03",

  "headline": "SkillBlend beat the naive average but not the single best model on this particular case.",

  "raw_sources": [{ "model": "GFS", "forecast_value": 306.02, "weight": null, "historical_skill": null }, ...],

  "naive_average": 269.11,

  "single_model_choice": { "model": "IFS", "forecast_value": 287.83 },

  "skillblend_blend": 289.49, "reference_value": 282.54,

  "naive_average_error": 13.43, "single_model_error": 5.29, "skillblend_error": 6.95,

  "narrative": [

    "Raw sources disagreed materially: GFS=306.0mm, IFS=287.8mm, AIFS=213.5mm for Bay of Bengal / East Coast at +120h lead.",

    "The rule-based regime classifier read this as 'depression' (context stability 54%).",

    "A naive equal-weight average would have given 269.1mm (13.4mm error against the verified 282.5mm).",

    "Always trusting IFS alone (its best overall historical RMSE for this region/variable) would have given 287.8mm (5.3mm error).",

    "SkillBlend's context-aware blend gave 289.5mm (7.0mm error) after bias correction."

  ]

}

GET /skill/verification response shape (array, one row per region/variable):

json

[{ "region": "KWG", "variable": "precipitation", "threshold": 50.0, "best_single_model": "AIFS", "best_single_model_csi": 0.75, "skillblend_csi": 0.8889, "relative_csi_improvement": 0.1852, "meets_target": true, "target_relative_csi_improvement": 0.05 }]

Build with realistic mock data matching these exact shapes for development so every screen looks fully alive immediately (including at least one mock case with weight: 0.0, one with abstain: true, and one honest-miss replay case), then wire to the live API base URL (a single configurable constant, defaulting to /api/v1).

Technical expectations

React + TypeScript, component-driven, clean state management.

Use Framer Motion for UI micro-interactions and page transitions, react-three-fiber + drei for the 3D hero, and GSAP with ScrollTrigger for scroll-choreographed sections if Framer Motion's scroll hooks aren't enough for the complexity described.

Charts: pick whatever renders the most polished, animated result (Recharts, visx, or custom SVG/D3 — your call). The weight-map lead-time curves and the region×variable verification matrix are the two charts worth the most craft.

Fully responsive, but the 3D/parallax hero should degrade gracefully (simplified motion, not broken) on mobile rather than being cut entirely.

Dark mode as the primary/default experience — this product lives in the dark.

Performance matters: the animations need to feel buttery, not janky — prefer GPU-accelerated transforms over layout-triggering properties throughout.    ### Implementation priorities and reliability rules

Build the application in this order:

1. Functional application shell, routing, typed API client, global state, loading/error/empty states.

2. Live Forecast Dashboard.

3. Weight Map Explorer.

4. Historical Replay / Counterfactual Replay.

5. Verification Scorecard.

6. Extreme Weather Guidance.

7. About / Architecture.

8. Landing-page 3D effects and advanced motion polish.

Create a single configurable `API_BASE_URL` and a typed API service layer. Never hard-code API calls inside visual components.

All displayed forecast, weight, trust, verification, replay, and extreme-weather values must come from API responses or realistic mock fixtures. Never hard-code scientific results into UI components.

Use realistic mock fixtures automatically when the backend is unavailable. The UI must remain fully functional in mock mode.

The lead-time selector must be populated dynamically from `GET /lead-times`.

Do not assume every API call succeeds. Implement proper loading, error, retry, empty-data, and abstention states.

Historical Replay Mode must be available globally and must work without live external weather-data access. It should be the reliable judging/demo path.

Treat `abstain: true`, `weight: 0`, negative verification improvement, unavailable data, and model disagreement as legitimate scientific states — never as frontend errors.

Prioritize scientific transparency and functional correctness over visual effects. 3D, particles, parallax, and animation must never block rendering, API functionality, accessibility, or mobile usability.

Do not invent missing backend fields. If a field is unavailable, handle it gracefully in the UI.

Keep components modular so the backend API contract can evolve without rewriting the visual layer.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/5ba3eaa1-697c-4fc8-894d-5e2c1c08f3a8).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
