# Project Context for Claude Code

Read [`README.md`](README.md) first and follow its index — overview, setup,
the reproduction path, and the benchmark live there.

Per-topic knowledge lives in `docs/` and applies to every contributor, not
just Claude; the session rules below are only routing:

- [`docs/development.md`](docs/development.md) — **the non-negotiables that
  gate every change** (proof-level byte gates, assemble-zorch's-blocks,
  lockstep pins, wheel provenance), plus witness-porting, `binary_field_ghash`
  dtype, and Pallas kernel gotchas. Read the non-negotiables before changing
  any prover behavior.
- [`docs/measurement.md`](docs/measurement.md) — read before quoting or
  comparing ANY benchmark number; the toolchain and methodology traps in it
  have each cost a session.
- [`docs/conventions.md`](docs/conventions.md) — how the protocol is modelled:
  claims, roles, and what the tooling misses.
