# Bounded one-shot delegation

A one-shot CLI request has no interactive completion consumer. Both ordinary `hermes chat --oneshot -q ...` and quiet `-Q` calls therefore return delegated results within the current tool call. Interactive sessions keep asynchronous completion delivery. This routing is a property of the CLI session and cannot be overridden by model arguments.

`--run-budget N` (or `agent.run_budget_seconds`) now interrupts active inference and attached children when the turn deadline expires, even while reasoning tokens continue arriving. The existing 80 percent wrap-up notice remains. Expiry returns a failed result and a nonzero CLI exit status. A completed turn cancels its deadline before returning, so it cannot interrupt a later turn. Time limits do not constitute aggregate token or dollar limits, nor a filesystem sandbox.

Verification uses real CLI subprocesses and an isolated local HTTP provider: successful child handback in normal and quiet modes, continuously active parent streams, and continuously active child streams. The same handback and cancellation checks fail against the prior runtime. Production providers are checked separately after the reviewed runtime is installed.

The surrounding legacy entrypoints remain large. This repair adds narrow routing and exit-status wiring; deadline ownership is isolated in `agent/run_deadline.py`. Broader CLI/facade decomposition remains separate maintenance, not part of this lifecycle fix.
