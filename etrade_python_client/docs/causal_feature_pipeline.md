# Causal Feature and Model-Preparation Contract

This contract covers the legacy HMM/EV research path in
`live_trading/data_ingestion.py` and `live_trading/ev_engine.py`. It does not
promote that path to order eligibility.

## Fit boundary

Every feature build requires an explicit `fit_end`. Feature eligibility,
missingness rejection, fractional-d selection, scaler policy, PCA dimension,
and HMM component selection may use only observations available through that
date.

- `train_regime_hmm(..., expanding_window=True)` is walk-forward mode. It
  requires `fit_end` to precede at least one evaluation row. PCA/HMM refits use
  rows strictly before the scored row and only the final filtered posterior is
  retained.
- `train_regime_hmm(..., expanding_window=False, fit_end=...)` is fixed
  out-of-sample mode when rows exist after `fit_end`. The scaler, PCA, and HMM
  are frozen from the fit prefix. Only rows after the cutoff are returned as
  an evaluable trace.
- Omitting `fit_end` is allowed only for the existing non-expanding,
  full-sample research fit. Its output is labeled
  `full_sample_research_only` and cannot feed the typed probability engine.

Daily-close outputs are available at `close_T_for_next_session`.

## Prefix invariants

Appending future rows must not change any earlier:

- selected feature column;
- fractional-d decision;
- stationary or scaled value;
- PCA component count or projected value;
- walk-forward/fixed-OOS HMM state or posterior;
- EV engine current-state input.

The implementation enforces this by selecting from the declared fit prefix,
using forward fill only, using trailing filters, and using a rolling scaler
whose window does not depend on request length. Backward fill, centered
windows, global evaluation-period scaling, and request-wide PCA fitting are
not permitted.

Each model stores a `CausalFeatureManifest` and
`CausalModelPreparationManifest`. The feature manifest binds the ordered,
selected training-prefix values as well as the transformation choices. The
model manifest binds the actual emission feature set, fit cutoff, PCA
dimension, HMM selection mode, and scaler mode. Cached fixed-snapshot models
must be replayed with their frozen prefix scaler; a rolling scaler is not an
equivalent transform. The probability engine consumes a persisted causal tail
posterior bound to those manifests. It does not fetch data or refit
preprocessing during probability construction. Missing or mismatched
provenance fails closed.

## Validation status

The contract implements and tests the causality boundaries in market-regime
specification Sections 2, 3, 10.6, 11.3, and 12.1–12.6. It remains
`UNVERIFIED` and `execution_eligible=false`.

Production certification is still blocked by independent verification of
upstream data completeness and decision-time availability, provider
entitlements, regime taxonomy/calibration, return-bucket provenance, model
registry/signing, and end-to-end live composition.
