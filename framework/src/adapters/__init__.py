"""Adapters: cross-domain estimate exchange (SPEC-006, plan baseline v0.1 §7/§12).

Hard boundary discipline: this package NEVER imports ortools / any solver /
VisitModel / VisitIR. The only channel towards the solver domain is the
EstimateEnvelope JSON contract (schemas/estimate-envelope/0.1.0) and the
sp_solve_ip-shaped kwargs dict assembled by :mod:`.pjp`. Solver round-trip
verification lives in tests, guarded by ``VISITMODEL_PATH``.
"""
