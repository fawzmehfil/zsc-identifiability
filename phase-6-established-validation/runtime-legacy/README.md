# Isolated Legacy Bridge

This file-protocol bridge is installed into two separate projects: Python 3.9
for the legacy ZSC-Eval asset audit, and Python 3.10 for the pinned ToMZSC
reference tooling required by its JAX 0.4.38 dependency. It verifies full
upstream hashes before execution. Missing official ZSC-Eval policy or
best-response assets produce `secondary_unavailable` rather than a fabricated
cross-check.
