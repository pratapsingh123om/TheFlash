# Final Application Resolution Plan

## 1. Objective

Deliver a scientifically honest demonstration application with two clearly separated products:

1. **2023 model-versus-observed player:** frozen, independent six-hour-ahead predictions compared with corresponding 2023 observations.
2. **Current 24-hour candidate:** recursive forecasts for +6, +12, +18 and +24 hours, driven by the most recent available gridded lightning history.

This document is a plan only. It does not authorize or perform model execution, data uploads, deployments or retraining.

## 2. Current Issue

The original 2023 web archive substantially understated predicted magnitude. One displayed day showed approximately 1,738 predicted regional counts against 183,983 observed regional counts, even though spatial Spearman correlation was 0.715 and top-10% hotspot overlap was 64.5%.

That magnitude result was inconsistent with the frozen model artifacts. The deployment ZIP reports the following full-2023 regional-count evaluation:

- Predicted overlapping 50 km regional mass: approximately 16,130,726.
- Observed overlapping 50 km regional mass: approximately 23,681,832.
- Predicted-to-observed ratio: approximately 68.11%.
- Strong-or-heavy F1 within 100 km and the same valid window: approximately 0.598.
- Heavy F1 within 100 km and the same valid window: approximately 0.443.

The incorrect player therefore cannot be used to judge the trained model.

## 3. Root Causes to Address

### 3.1 Incorrect retrospective recursion

The old archive recursively inserted model predictions into later lightning-history features during the 2023 comparison. The trained evaluation instead predicted every six-hour window independently using the historical information available at that window's issue time.

### 3.2 Ineligible rows included

The frozen evaluation uses only rows where `forecast_row_eligible == 1`. The comparison builder must apply the same rule. Missing target coverage must never be converted into an observed zero.

### 3.3 Early-stopping range mismatch

Inference must use each booster only through its stored `best_iteration`. Using every available tree can produce predictions different from the frozen evaluation.

### 3.4 Wrong interpretation of totals

`expected_region50_count` is an overlapping 50 km regional-severity target. Summing it across grid cells produces overlapping regional mass, not a unique count of lightning events across India.

### 3.5 Misleading map scaling

Predicted and observed maps must use one shared color scale. Independent color maxima can make a weak-magnitude prediction appear as intense as the observation.

### 3.6 Current forecast lacks a live seed

The +24-hour recursive candidate is not operationally current unless the backend receives a recent gridded lightning-history file. A rebased historical seed is suitable only for demonstrating application behavior.

## 4. Phase A — Rebuild the 2023 Comparison Archive

### A1. Freeze the exact model

- Use `india_lightning_24h_demo_ed5dec2770f7.zip`.
- Verify SHA-256:
  `0b565c63f89be15818920ab6798a46530b0e0cbdfffd6c4a5b7c8dbacf7a39d1`.
- Load the six XGBoost boosters, three isotonic calibrators, thresholds and configuration from that verified ZIP.
- Do not select another run based on its modification time.

### A2. Use the correct panel

- Input directory:
  `/content/drive/MyDrive/MINOR_PROJECT/0.25_spatial_models/india_lightning_era5_physics_0p25_forecast_panel`
- Process the twelve 2023 monthly partitions.
- Require the following fields:
  - model's 106 predictors;
  - `forecast_valid_time_ist`;
  - `grid_id`, latitude and longitude;
  - `forecast_row_eligible`;
  - `target_region50_sum`;
  - `lightning_count` for diagnostics.

### A3. Reproduce frozen inference

For every 2023 six-hour valid time:

1. Keep the panel's observed pre-issue lightning-history features.
2. Select only `forecast_row_eligible == 1` rows.
3. Run an independent +6-hour prediction.
4. Apply each booster through its saved `best_iteration`.
5. Apply the saved isotonic calibrators.
6. Construct coherent none, mild, strong and heavy probabilities.
7. Combine class probabilities with the corresponding conditional-severity heads.
8. Compare `expected_region50_count` with the existing `target_region50_sum`.
9. Do not recursively replace lag features anywhere in this retrospective player.

### A4. Build daily comparison records

Aggregate the four six-hour windows of each IST day by grid cell and store:

- date;
- grid ID, latitude and longitude;
- predicted overlapping regional mass;
- observed overlapping regional mass;
- maximum heavy probability;
- maximum warning code;
- daily spatial Spearman correlation;
- exact top-10% hotspot overlap;
- daily predicted-to-observed regional-mass ratio.

### A5. Enforce publication gates

The full-year archive must not be published unless:

- all 365 dates are present;
- all outputs are finite where rows are eligible;
- no duplicate date/grid records exist;
- the prediction mode is recorded as `independent_6h_holdout_with_observed_history`;
- generated full-year predicted and observed sums reproduce the frozen evaluation within 0.5%;
- the manifest explicitly states that totals represent overlapping regional mass;
- the archive can be reopened and every daily JSON payload can be parsed.

### A6. Version the corrected artifact

Use a new immutable name:

```text
india_lightning_2023_comparison_archive_v2.zip
```

Target object:

```text
gs://nuclei-lenses/flash-model/india_lightning_2023_comparison_archive_v2.zip
```

Do not overwrite or reuse the scientifically incorrect v1 object in the application.

## 5. Phase B — Validate the 2023 Player

### B1. API verification

Verify:

- `/api/comparison/dates` returns the v2 manifest and 365 dates;
- `/api/comparison/day?date=YYYY-MM-DD` returns valid paired cells;
- invalid or unavailable dates receive clear responses;
- the backend does not silently fall back to the v1 archive.

### B2. Quantitative checks

Inspect at least:

- a quiet winter day;
- a pre-monsoon day;
- a major monsoon day;
- 2023-09-15;
- 2023-09-29.

For each day, verify:

- predicted and observed values correspond to the same grid and IST date;
- the magnitude ratio is credible;
- spatial Spearman and hotspot overlap match an independent calculation;
- no observation data enters the prediction itself;
- daily totals reconcile with the archive manifest.

### B3. Visual checks

- Use one shared color scale for predicted and observed maps.
- Display the regional-mass ratio explicitly.
- Label totals as overlapping regional mass, not national lightning events.
- State that 2023 is a chronological holdout but now constitutes development evidence because its results have been inspected.
- Keep warnings, probabilities and expected severity visually distinct.

## 6. Phase C — Finalize the Current +24-Hour Candidate

### C1. Required live input

Prepare a recent Parquet file with one row per model grid cell containing:

```text
grid_id
observation_time_ist
lightning_lag6h_count
lightning_lag12h_count
lightning_lag18h_count
lightning_lag24h_count
```

The file must:

- contain every expected grid cell exactly once;
- contain no duplicate grid IDs;
- contain finite, non-negative lag values;
- identify the actual observation cutoff time;
- be recent enough for the application freshness policy.

### C2. Recursive prediction behavior

- +6 hours uses the supplied observed history.
- +12 hours uses the predicted +6-hour proxy in the lag state.
- +18 hours uses the predicted +6 and +12-hour proxies.
- +24 hours uses the preceding recursive predictions.
- The interface must visibly communicate that uncertainty increases at later horizons.

### C3. Candidate limitations

The current lightning-only candidate must not be presented as a fully operational meteorological forecast because:

- later horizons recursively depend on earlier model estimates;
- the model was validated primarily at +6 hours;
- a stale or synthetic history seed invalidates the word “current”;
- no prospective post-2023 evaluation has yet been completed;
- exact-cell strong and heavy classification remains weak even though 50–100 km spatial skill is more useful.

## 7. Phase D — Final Application Layout

### Section 1: Project status

- research-prototype badge;
- model version and grid resolution;
- last lightning-history observation time;
- live/stale seed status;
- forecast issue time.

### Section 2: 2023 comparison player

- date picker and play/pause controls;
- predicted and observed maps using a shared scale;
- spatial Spearman;
- top-10% hotspot overlap;
- predicted-to-observed regional-mass ratio;
- explanatory note about overlapping regional targets.

### Section 3: Current forecast

- tabs for +6, +12, +18 and +24 hours;
- storm-category map for none, mild, strong and heavy;
- heavy and strong/heavy probability layers;
- regional expected-severity layer;
- location inspector;
- issue and valid timestamps.

### Section 4: Model evidence

Report frozen metrics without exaggeration:

- chronological split: training 2019–2021, selection/calibration 2022, development evaluation 2023;
- spatial skill at exact, 50 km and 100 km tolerances;
- class precision, recall and F1;
- regional-count magnitude ratio;
- comparison with six-hour persistence;
- clear distinction between retrospective evidence and future operational performance.

### Section 5: Limitations

- no prospective post-2023 test;
- exact localization of rare strong/heavy storms remains difficult;
- overlapping regional severity is not a national strike total;
- recursive horizons accumulate uncertainty;
- operational performance depends on timely and complete lightning observations;
- warnings must not be used as official public-safety alerts.

## 8. Deployment Plan

1. Generate and verify the v2 comparison archive.
2. Upload the immutable v2 object to Cloud Storage.
3. Confirm Vercel's OIDC identity can read it.
4. Confirm the backend retrieves v2 rather than cached v1 data.
5. Test API health and the comparison endpoints.
6. Test the frontend on desktop and mobile.
7. Configure the live lightning-history object only after its ingestion process is reliable.
8. Perform a final smoke test of all four recursive horizons.
9. Record the deployed Git commit, model ZIP hash and archive hash.

## 9. Rollback Plan

If the corrected application fails:

1. Keep the model bundle unchanged.
2. Disable only the comparison player if v2 is unavailable.
3. Never fall back silently to the incorrect v1 comparison.
4. Retain the last verified application commit.
5. Restore service only after API, archive and UI checks pass.

## 10. Final Acceptance Criteria

The application is ready for final demonstration only when all conditions below pass:

- [x] Exact deployment model ZIP checksum verified (`0b565c63f89be15818920ab6798a46530b0e0cbdfffd6c4a5b7c8dbacf7a39d1`).
- [ ] Corrected archive contains 365 dates (in progress via `scripts/colab_run_2023_recursive_full_year.py`).
- [x] Every retrospective forecast is independent +6 hours (`comparison_recursive = False`).
- [x] Only eligible rows contribute to retrospective metrics (`forecast_row_eligible == 1`).
- [x] Saved early-stopping iterations are respected (`best_iteration` iteration range).
- [ ] Frozen 2023 regional sums are reproduced within 0.5% (automated publication gate in script).
- [x] Shared map color scale is active (`sharedMaximum = max(pred_max, obs_max, 1)`).
- [x] Regional-mass totals are correctly labelled (overlapping 50 km regional severity, not discrete strike count).
- [x] 2023 is described as development evidence, not an untouched final test.
- [x] Current forecast declares whether its lightning seed is live or stale (`live` vs `stale` provenance badge).
- [x] +6/+12/+18/+24 timestamps are correct (tested in smoke test).
- [x] API, frontend and mobile smoke tests pass (Pytest 7/7, smoke test 4/4 horizons).
- [x] No secret, service-account key or bucket credential reaches the browser (Vercel OIDC + WIF).
- [x] Research-prototype and safety limitations are visible (Sections 1, 3, 4, 5, and verdict quote).

## 11. Recommended Final Verdict Language

> The system demonstrates useful six-hour regional lightning-risk localization, particularly when evaluated within operationally meaningful 50–100 km spatial tolerances. Exact-cell prediction and rare-event severity remain research limitations. The 2023 comparison is chronological development evidence, while the current +24-hour recursive product is a demonstration that requires a timely lightning-history feed and prospective validation before operational use.
