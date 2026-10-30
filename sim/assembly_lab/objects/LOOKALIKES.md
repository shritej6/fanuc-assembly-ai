# Look-alike objects: where they differ and which camera can see it

Agent 2 (T12), 2026-10-30. Every number here was computed from the printed STLs (`cad/stl`) and the cell calibration
(`robot/cell/calibration/cameras.json`). Nothing was photographed. The visibility and depth results are **PRELIMINARY**
(geometry only, no lighting or contrast model) until a wrist test on the real parts confirms them.

How to regenerate:
- `PYTHONPATH=. uv run python sim/assembly_lab/objects/build_objects.py` writes `objects.yaml`, which holds
  `distinguishing_dims_mm`, `lookalike_groups` and `size_twins_within_1mm`.
- `PYTHONPATH=. uv run python sim/assembly_lab/objects/visibility.py` writes `visibility.yaml` (the ray-cast). It takes
  about 9 minutes.

## 1. Are F01 = F02 and F04 = F05 = F06 really identical? Yes (measured)

| Object | Triangles | sha256 of the STL body (first 16 hex) | Volume mm³ | Bbox mm | Centre of mass mm |
|---|---|---|---|---|---|
| F01 | 3904 | 3e33719bda008cb2 | 23818.0 | [-30,-25,-8]..[30,25,3] | (-0.007, -0.123, -3.911) |
| F02 | 3904 | **3e33719bda008cb2** | 23818.0 | same | (-0.007, -0.123, -3.911) |
| F03 | 3904 | dca879d5b1627795 | 23818.0 | same | (-0.007, -0.123, -3.911) |
| F04 | 3904 | 50550f1d632e6d79 | 23818.0 | same | (-0.007, -0.113, -3.911) |
| F05 | 3904 | **50550f1d632e6d79** | 23818.0 | same | (-0.007, -0.113, -3.911) |
| F06 | 3904 | **50550f1d632e6d79** | 23818.0 | same | (-0.007, -0.113, -3.911) |

**How the pairs were compared.** Triangle sets were compared after rounding vertices to 1e-5 mm and sorting. The
Hausdorff distance is two-way and nearest-neighbour, over about 14.7 million surface samples per mesh at 0.15 mm
spacing.

| Pair | Same triangles | Hausdorff mm | What differs |
|---|---|---|---|
| F01~F02 | yes | 0.000 | Nothing. The STL bodies are byte-identical. |
| F04~F05, F04~F06, F05~F06 | yes | 0.000 | Nothing. The STL bodies are byte-identical. |
| F01~F03 | no | 1.001 | The housing stops are 1 mm further out on each side (see below). |
| F03~F04 | no | 1.414 | The side stops are 1 mm further out; the front L-stops are 1 mm further back. |
| F01~F04 | no | 2.236 | The side stops are 2 mm further out; the front L-stops are 1 mm further back. |

The six bases differ only in where the stops stand around the housing footprint (`fixtures.make_fixture(housing_x,
housing_y)`). The pegs, plate and holes are the same on every base.

| Base type | Bases | Housing opening between the stops (x × y) | Side-stop centres | Front L-stop inner corner |
|---|---|---|---|---|
| A | F01, F02 | 35 × 33 | x ±19, y 10 | (±17.5, -16.5) |
| B | F03 | 37 × 33 | x ±20 | (±18.5, -16.5) |
| C | F04, F05, F06 | 39 × 31 | x ±21 | (±19.5, -15.5) |

So F01 and F02 cannot be told apart by any sensor, and neither can F04, F05 and F06. They do not need to be: any
base of a type serves every product of that type (PLAN 3.2).

## 2. Per group: the distinguishing feature, the best view and a fallback

**How visibility was tested.** Each part stands on the table in its installed orientation at UF0 (400, 0), at 8
yaws. A ray-cast with self-occlusion checks each camera. A pair counts as *visible* when, at the worst yaw, the
differing geometry projects to at least 12 px² and the smallest cue spans at least 2 px.

**Camera sampling at the part:**

| Camera | mm per px | Notes |
|---|---|---|
| Table | 0.86 | |
| Side | 0.55 | |
| Wrist | 0.365 | At a 150 mm standoff. The wrist camera captures at 848×480, so fx is 410.5; the 1280 image is upscaled. |

**Columns in the table below:**
- **Cue** is the smallest dimension that has to be resolved (`objects.yaml` → `best_cue_mm`).
- **px** is that cue's size in pixels, given for the table, side and wrist cameras in that order.

| Group | Feature (part-local mm) | Cue | Table / side / wrist px | Visible table · side · wrist | Best view | Fallback if missed |
|---|---|---|---|---|---|---|
| F01/F02 vs F03 | Side stops at x ±19 vs ±20; L-stops at x ±17.5 vs ±18.5 (3 mm tall) | 1.38 | 1.6 / 2.5 / 3.8 | no · yes · yes | Wrist top-down at 150 mm | **Partial.** An M3_01 housing (36 wide) will not enter an A base (35 opening), so the seated-height check catches it. An M1/M2 housing on F03 seats with 1.5 mm play per side and is **not** caught, so the base must be identified. |
| F01-03 vs F04-06 | Stops 2 mm wider in x and 1 mm further back in y | 1.6-3.0 | 1.9-3.5 / 3-5.5 / 4.4-8.2 | yes (A vs C) · yes · yes | Wrist top-down, or the table camera for A vs C | **Yes.** A connector housing (38×30) does not fit A or B. A control housing (y 32) does not fit C (opening 31). |
| M4_03 / M5_03 / M6_03 guides | 3 × 2 mm through-notch at one outer corner: M4 (-x,-y), M5 (+x,+y), M6 (+x,-y), through the full 5 mm frame (z 0-5) | 1.67-1.75 | 1.9-2.0 / 3.1-3.2 / 4.6-4.8 | marginal (M4~M6 and M5~M6: no) · M4~M5 only · **yes** | Wrist 30° oblique toward the notch corner (best: from the front for M4~M6, 188 px²) | **None.** The notch is cosmetic: every guide fits every connector housing, and every lock fits every guide. Identity has to come from the camera. |
| M4_04 / M5_04 / M6_04 locks | Bridge-top detail at local y -3..+3, 1 mm high: M4 has two ribs at x ±9 (1 × 5 mm); M5 has one rib at x -9 turned 40°; M6 has a 1 mm-deep chevron cut into the pad top | 1.0 | 1.2 / 1.8 / 2.7 | no · no · **yes** | Wrist top-down at ≤ 150 mm (2.7 px across a rib); oblique 30° from the left gives slightly more (143 px²) | **None.** Any lock fits any guide. |
| M5_02 / M6_02 inserts | Top markers 0.8 mm high (M5: two lines at x -11, -9; M6: a bar at y -8.5). Front pockets: M5 has six 3.2 × 3, M6 has two 9 × 8 | 0.8 (top) / 4+ (pockets) | 0.9 / 1.5 / 2.2 | no · no · **yes** | Wrist 30° oblique **from the front** (1038 px²: the pockets dominate) | **None.** The shared envelope is deliberate (`model_05` docstring: the wrong-part recovery test). Either insert fits either housing. |
| M1_02 / M2_02 carriers | Key post 6 mm (M1) vs 5 mm (M2) tall (7 × 10 top, 70 mm²); M2 also has two 0.6 mm flange grooves | 1.0 | 1.2 / 1.8 / 2.7 | no · no · **yes** | Wrist top-down. The post height is a 1 mm depth step over 70 mm², read against the flange of the same part. | **Partial.** Cap M2_03 (5.5 mm socket) on post M1_02 rides 0.5 mm high, which fails the ±0.3 mm seated-height gate. Knob M1_03 on M2_02 seats fully (a deliberate wrong-part test), so it is **not** caught. |
| M1_01 / M3_01 housings (**not in a taxonomy group**) | Pocket Ø19 round (M1) vs 17 × 13 rectangle with a key block (M3). M1 also has 1 mm side ribs. The bboxes are equal (36 × 49 × 16). | 3.39 | 3.9 / 6.2 / 9.3 | yes · yes · yes | Any camera; wrist top-down is unambiguous | **Yes.** M3_01 does not fit an A base (36 > 35). M1_01 on F03 seats loosely and is not caught, but the carrier and stem do not fit the wrong pocket shape. |
| M1_01 / M2_01 housings | Side ribs 1 × 2 × 8 mm at x ±17.5 (M1). Three front ribs and a 3 × 2 notch at the front-left top (M2). | 2.0 | 2.3 / 3.7 / 5.5 | yes (worst yaw 13.9 px²) · yes · yes | Wrist 30° oblique from the front (246 px²) | **None.** The same carrier, cap and guard fit both. |
| M4_01 / M5_01 / M6_01 housings | Cavity 25 × 19 (M4, with two guide ribs) vs 27 × 21 (M5/M6). M5 vs M6 differ only in rib layout (rear rib pair at x 8 / 11 on M5; four left ribs on M6). | 1.8-3.2 | 2.1-3.6 / 3.3-5.8 / 4.9-8.7 | yes · yes · yes | M4 vs others: table camera is enough. M5 vs M6: wrist 30° oblique from the left (655 px²). | **Partial.** M5/M6_02 do not fit an M4 cavity. M4_02 in an M5/M6 cavity seats with 1.5 mm play (not caught by height). |
| M1_04 / M2_04 / M3_04 guards | Bridge width 25 / 27 / 29 mm. Notch shapes: M1 right corner, M2 left end plus R1.5 corners, M3 centre triangle. | 2.0-4.0 | 2.3-4.6 / 3.7-7.3 / 5.5-11 | yes · yes · yes | Wrist oblique from the right (largest area) | **Partial.** The arm spacing is 20 / 20 / 22 mm, so M3_04 does not enter an M1/M2 housing. M1_04 and M2_04 are interchangeable. |
| M1_02 / M2_03 (**size twins**, different families) | From above both show a Ø22 disc with a 7 × 10 block on top. Heights differ by 1 mm: flange 11 vs cap 10, post 17 vs pad 18. From the side, the carrier has a Ø18 plug under a 2 mm-thick Ø22 flange; the cap is a solid Ø22 × 10 disc. | 1.0 (top) / 2.0 (side step) | | wrist yes (CAD dimensions; not ray-cast) | Wrist oblique: the side silhouette shows the 2 mm flange step | Sheet context: step 02 vs step 03 |

**Conclusion for training photos.** Every non-identical pair is geometrically visible to the wrist close-up at 150 mm.
Three groups are visible **only** to the wrist close-up and have **no** mechanical fallback, so new wrist training
photos help most here:
- guides M4/5/6_03;
- locks M4/5/6_04;
- inserts M5/M6_02.

The table camera cannot separate the locks, the M5/M6 inserts, M1_02/M2_02, or the M4~M6 / M5~M6 guides. The side
camera is no better for those.

## 3. Can the wrist close-up resolve a 0.8-1.0 mm cue? (PRELIMINARY)

**Calibration facts** (`cameras.json`, `FRAMES.md`):
- Wrist depth is 848×480 with fx = fy = 416.85 px (HFOV 91°).
- Wrist colour is fx 619.6 at 1280×720, which also gives HFOV 92°. It is captured at 848×480, so the true sampling is
  fx 410.5.
- Both lenses sit on the flange 163-175 mm along the tool and about 98 mm beside it, looking along the tool.
- With the fingers straight down, the lens is about 118 mm (depth) or 130 mm (colour) above the fingertip plane, and
  about 98 mm to the side of the fingers.

**Sensor class (inferred, not confirmed).** The colour HFOV equals the depth HFOV (92°). A RealSense D435 colour
camera is 69°, while a D405 takes colour from the same imager as depth (87° nominal), so this is most likely a
**D405-class** camera. That means:
- an 18 mm stereo baseline;
- an ideal range of 7-50 cm;
- no IR projector, so passive stereo.

Source: Intel RealSense D400 series datasheet, rev. 2023+.

**Depth noise model.** For active and passive stereo, RMS depth noise = Z² · subpixel / (f · baseline). Source:
Grunnet-Jepsen, Sweetser, Woodfill, "Tuning depth cameras for best performance", Intel RealSense white paper, which
gives a typical subpixel of 0.08.

| Standoff Z (lens to surface) | Colour mm/px (fx 410.5) | Depth px pitch | Depth RMS noise, D405-class theory |
|---|---|---|---|
| 130 mm (colour lens; fingertips level with the part top) | 0.32 | 0.31 | 0.18 mm |
| 150 mm (proposed) | 0.37 | 0.36 | 0.24 mm |
| 200 mm | 0.49 | 0.48 | 0.43 mm |
| 400 mm (the calibration pose) | 0.97 | 0.96 | 1.7 mm. The measured calibration plane residual there is `rms_plane_mm` 3.0, which includes calibration error. |

**Answers:**
- **Pixels.** At 150 mm a 1.0 mm cue spans 2.7 px and a 0.8 mm cue spans 2.2 px. That is resolvable for a learned
  classifier if the edge has contrast, but black on black is weak. 130 mm gives 3.2 and 2.5 px.
- **Depth.** A 1 mm step is about 4σ per pixel at 150 mm (theory). Over the 70 mm² carrier post top (about 500 px) the
  mean height resolves well below 0.1 mm *if the black PLA returns depth at all*. Passive stereo on texture-poor black
  plastic often leaves holes. Use depth only for the M1_02/M2_02 post height and seated-height checks, and treat
  missing depth as "unsure" (second view, then Andon; ROBOT_BRAIN §3).
- Thin features (the 1 mm lock ribs, the 0.8 mm insert markers) are narrower than the stereo matching window, so depth
  will smooth them away. Those pairs must be read from colour.

**Proposed close-up** (the planner can derive the TCP from these):
- Stand off 150 mm from the colour lens to the part's top centre. The fingertips are then 20 mm above the part top
  (130 mm lens-to-tip) and about 98 mm to the side. The planner must check that spot for finger clearance like any
  other finger position.
- Locks, carriers and the M4~M5 guides: top-down.
- Guides M4~M6, inserts M5/M6_02, housings M1/M2_01 and M5/M6_01: a second view, 30° oblique from the side the table
  above names. The yaw is known, so the side is chosen in the part frame.
- One top-down plus one oblique view covers every pair. That is the "second close-up from another angle" of ROBOT_BRAIN
  §3.

**To confirm on the cell (cannot be computed):**
- the wrist camera model;
- whether black PLA returns depth at 150 mm;
- whether the 1 mm lock ribs and 0.8 mm markers show in colour under the cell lighting.
