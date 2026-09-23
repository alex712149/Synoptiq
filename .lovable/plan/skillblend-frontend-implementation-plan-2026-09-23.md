# SkillBlend frontend implementation plan

## Goal
Build a complete, judge-ready weather-intelligence application around the uploaded SkillBlend API contract. The application will default to the live API at `/api/v1`, automatically fall back to realistic typed fixtures when it is unavailable, and keep Historical Replay dependable as the primary demo path.

## Experience and visual direction
- Create a dark atmospheric command-center system using storm-cyan, warning amber, deep atmospheric blue-violet, technical display typography, restrained glass surfaces, fine map/radar grids, and a distinct calm “scientifically honest” treatment for misses.
- Use a shared responsive shell with desktop sidebar, compact mobile navigation, live/mock health indicator, global Historical Replay access, consistent loading/error/empty states, and accessible focus/motion behavior.
- Keep scientific content readable before effects load. Honor reduced motion, simplify the globe and parallax on smaller devices, and never let animation block controls or data.

## Data foundation
- Define exact TypeScript models for regions, variables, sources, forecasts, weight maps, verification, extreme guidance, and replay responses.
- Add one configurable `API_BASE_URL`, a fetch/validation service, normalized API errors, retries, and mock fallback behavior outside visual components.
- Build realistic deterministic fixtures matching the uploaded backend exactly, including zero model weight, an abstention case, negative verification outcomes, calibrated/uncalibrated threats, and both clear-win and honest-miss replay events.
- Fetch the lead-time ladder dynamically from `/lead-times`; fixture values are used only when fallback mode is active.
- Keep filters in route search parameters where shareable and use TanStack Query for caching/refetching.

## Routes and screens
1. **`/` — Landing observatory**
   - Full-bleed interactive React Three Fiber atmospheric Earth with storm bands, particles, subtle pointer tilt, local lighting, and a non-WebGL visual fallback.
   - Scroll choreography for model disagreement, animated three-source convergence, capability previews, proof matrix, honest-miss proof, and a “systems ready” transition into the application.
   - Ambient radar/grid depth, count-up readouts, physical button/card interactions, and reduced-motion alternatives.

2. **`/forecast` — Live Forecast Dashboard**
   - MapLibre view using bundled geographic data for the three real pilot coordinates, with selectable KWG, BOB, and IGP zones.
   - Region, variable, and API-driven lead-time controls.
   - Forecast pipeline, source weights/skill/value comparison, explicit zero-weight state, five-component trust instrument, regime probability spread, disagreement/bust state, abstention review state, explanation decomposition, and provenance.

3. **`/weights` — Weight Map Explorer**
   - Region, variable, season, and regime controls.
   - Animated full-ladder model-weight curves/areas plus synchronized trust decay chart and readable point details.

4. **`/replay` — Counterfactual Bust Replay**
   - Globally accessible replay event list with calm visual distinction between clear wins and honest misses.
   - Cinematic detail console with sequential truth-line comparison, raw-source context, error comparison, and timed narrative steps; controls remain usable without animation.

5. **`/verification` — Verification Scorecard**
   - Full region × variable CSI improvement heatmap, target marker, best-model context, clear rainfall wins, and visible negative outcomes.

6. **`/extremes` — Extreme Weather Guidance**
   - Region and API-driven lead selection with three threat instruments, escalating severity, and explicit calibrated/uncalibrated status.

7. **`/architecture` — About / Architecture**
   - Scroll-built pipeline diagram from source forecasts through trust, bust risk, and blended output, plus concise data provenance and documented prototype limitations.

## Technical implementation
- Install React Three Fiber, drei, Three.js, Motion, GSAP, MapLibre GL, and supporting typings compatible with React 19.
- Use modular feature folders for shell, shared instruments, each screen, data hooks, fixtures, charts, map, and 3D scene.
- Keep the R3F scene client-only without disabling server rendering for the entire landing page. Use local/procedural globe textures and no runtime CDN environment assets.
- Use Recharts/custom SVG for charts where it gives better label and animation control; use semantic design tokens from the global stylesheet throughout.
- Add unique title, description, Open Graph, and Twitter metadata for every content route.

## Validation
- Check the generated application for current build/runtime errors.
- Verify desktop and mobile layouts, navigation, live-to-mock fallback, retry behavior, dynamic lead times, all legitimate scientific edge states, and reduced-motion behavior.
- Use browser screenshots to confirm the 3D scene is visibly lit and framed, the map and charts render, no content overlaps, and the replay flow works end-to-end.
