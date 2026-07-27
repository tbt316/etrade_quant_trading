# Regime Data Provider Entitlement Gate

**Review date:** 2026-07-26

**Status:** unresolved production release gate

**Scope:** raw SPY and VIX inputs retained for regime detection

This document is an engineering release gate, not legal advice. The repository
must not assume that possession of an API key or access to a public CSV grants
permission to retain the response or use it in an investment strategy.

## Current provider review

The R2 evidence adapter targets Massive's SPY Daily Ticker Summary endpoint and
Cboe's VIX history CSV because they provide a clear regular-session close and a
first-party VIX source. The adapter is library-only: it is not scheduled, does
not run at import time, and is not connected to E*TRADE orders.

The current public terms do not establish the rights required for production
use:

- Massive's [Market Data Terms of Service](https://massive.com/legal/market-data-terms-of-service)
  say that data is generally for personal, non-business use and, unless a
  separate agreement applies, display use only. They specifically restrict
  non-display use and the creation of an investment strategy without an
  appropriate license.
- Massive's [business terms](https://massive.com/legal/businesses-terms-of-service)
  allow broader internal processing under an order form, but still require the
  customer to obtain applicable third-party rights and separately restrict
  creation of an investment strategy unless licensed.
- Cboe's [website terms](https://www.cboe.com/terms) limit ordinary website
  material use and restrict electronic storage and derivative uses without
  consent. Cboe's [content-permission page](https://www.cboe.com/use-of-content)
  says approval and an executed license are required for use of Cboe content.

Consequently, an individual Massive subscription and an anonymously accessible
Cboe CSV are not sufficient evidence for this system's non-display, retained,
strategy-derived use.

## Required production evidence

Before a live or continuously scheduled collector is enabled, the operator must
record all of the following outside the repository:

1. The subscriber and legal entity covered by each agreement.
2. The provider, dataset, instruments, account classification, and agreement or
   order-form identifier.
3. Explicit permission for non-display algorithmic use, local retention of
   exact response bytes, derived regime signals, and the intended number of
   users/accounts.
4. Retention, deletion-on-termination, redistribution, and display limits.
5. Effective and expiration dates, plus the person and date that reviewed the
   entitlement.

Production configuration must refer to that entitlement record by opaque ID.
It must not contain contract documents, credentials, API keys, or bearer tokens.
An expired, missing, or scope-mismatched entitlement fails closed before the
first network request.

## Data-handling rules

- Raw provider bytes remain in the local evidence database with restrictive
  permissions. They are never committed, logged, returned by a dashboard API,
  embedded in HTML, or copied into test fixtures.
- Unit and CI tests use synthetic provider-shaped payloads only.
- Fetch receipts contain a secret-free endpoint and request parameters; they
  never contain an Authorization header or API key.
- Provider termination or a deletion requirement must have an operator-reviewed
  purge procedure. Content-addressed deduplication does not override contractual
  deletion duties.
- A replacement provider must satisfy the same causal clock, finality, revision,
  and raw-lineage contract; provider access alone does not waive this gate.

Until these items are complete, provider-backed results remain research/shadow
artifacts and cannot influence order eligibility.
