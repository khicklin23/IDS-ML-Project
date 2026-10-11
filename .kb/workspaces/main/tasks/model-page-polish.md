---
id: 01a12b60-d425-71e3-b727-4a531424b189
slug: tasks/model-page-polish
title: "Simplify Model explained page + polish diagram/charts"
type: task
status: completed
priority: medium
agent_id: agent-2-env-setup
---

User-directed, 2026-10-11. Scope: ui/model.html, ui/model.js, ui/styles.css only (same files tasks/model-explanation-ui built). (1) Simplify the page's language further -- plain wording, jargon terms (backpropagation, ReLU, softmax, dropout, Adam, logits, sigmoid, inverse class frequency) get a short plain-words explanation in parentheses the first time each appears, matching the style just used in README.md's new 'Model versions' section. (2) Visual polish on the network diagram at the top (the procedurally-drawn SVG in model.js renderNetwork) -- user explicitly likes this one and wants more attention on it. (3) User reports 'the charts look pretty bad scrolling down' -- the scenario-example SVG sketches, the class-weight balance-blocks visual, and the threshold-visual bar are the candidates; apply the dataviz skill's mark specs (thin/consistent marks, proper spacing, legible labels, no flat ugly fills) since no browser tool exists here to see the current rendering directly. No backend changes. Coordinate: this is agent-1-static-ui's file set from tasks/model-explanation-ui/tasks/static-dashboard-ui (both completed) -- claiming here so work doesn't collide if that agent resumes.

### Done, 2026-10-11 (agent-2-env-setup)

**Network diagram** (`model.js::renderNetwork`): previously only the selected layer's *dots* changed color; all connection lines stayed flat gray regardless of selection. Now connections touching the selected layer (both incoming and outgoing) light up too (`#9bbdda` vs `#e6ebef`), so picking a layer shows where its signal comes from and where it goes, not just which dots it is. Active-layer dots got slightly bigger/bolder too. Updated the diagram caption to say so.

**Real bug found and fixed in the threshold visual**: the "normal" zone's width was hardcoded to CSS `50%` regardless of the actual saved threshold -- correct-looking purely by coincidence since this report's threshold happens to be 0.5, but wrong in general and actively misleading if it weren't. Now `model.js` sets the zone width to match the real threshold, same value driving the marker position. Restyled as a proper two-tone (green/warm) diverging bar with `0.0`/`1.0` tick labels, since that's exactly what a threshold cutoff is.

**Scenario-example sketches**: were drawn from raw, un-normalized pixel heights, so some cards (e.g. benign, max height 25) looked visually "flatter" than others (e.g. outbound_transfer, max height 40) for no real reason -- these are illustrative shapes, not a shared scale, so that inconsistency was just noise. Now each sketch normalizes to its own tallest bar; bars also got rounded tops (`rx="2"`) instead of flat-edged rects.

**Class-weight "balance" visual**: recolored off two ad-hoc hues (slate-blue/dull-purple, unrelated to anything else on the page) onto the site's own existing tokens -- a blue consistent with the page's accent color for the common class, the same amber already used for default-severity badges for the rare class (ties into the existing "this needs attention" visual language instead of inventing a new one). Added a baseline rule under the blocks.

**Language simplification**: added short parenthetical, plain-words definitions the first time each jargon term appears -- backpropagation, Adam, ReLU, dropout (plus the actual percentage used), logits, softmax, sigmoid, bias, checkpoint, "nonlinear," and the multiclass loss-weighting explanation (replaced the literal "inversely to class frequency" with a plain-language description, instead of just parenthesizing it). Left the explicitly-opt-in `<details>` "equations" aside more technical, consistent with it being flagged as a deeper dive already.

**Verification**: no browser tool exists here, so verified with `npm run check` (`node --check` on all three JS files + the project's `node:test` suite) -- this caught a real regression immediately: two existing tests asserted the literal old wording `Linear(1)`/`Linear(6)` in the architecture-details text, which my simplification changed to `(1 output)`/`(6 outputs)`. Updated `checks/model.test.mjs`'s two regexes to match the new (still layer-size-verifying) text rather than loosening or deleting the assertions. All 23 checks pass. Also verified via a local `python3 -m http.server` that `model.html`/`model.js`/`styles.css` still serve correctly. Actual visual appearance in a real browser is still unverified -- same limitation every UI pass in this project has had.