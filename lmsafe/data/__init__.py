"""Data layer: turns raw authentication-event logs into labeled,
time-windowed graphs that the rest of lmsafe consumes.

Sub-modules
-----------
schema        canonical in-memory Event representation shared by every
              data source (LANL, OpTC, synthetic), so detectors/policies/
              metrics never need to know which dataset they are looking at.
lanl_stream   low-level, dependency-free streaming reader for the LANL
              cyber1 .txt.gz files (never downloads the whole file).
lanl.py       the fair LANL extraction protocol built on top of
              lanl_stream: temporal split, red-team label join, remote-
              LogOn filter. This is the module scripts/ actually calls.
optc.py       OpTC schema adapter (Phase-2 dataset, used for the transfer
              check in a later chapter -- stubbed with a clear TODO path).
windows.py    groups a stream of Events into fixed-length time windows and
              builds the per-window graph (node index + weighted edge list)
              that lmsafe.detectors consumes.
"""
