# Realism check

20 unselected synthetic matches (both teams combined) against the Wyscout public dataset.

**Real reference not generated yet.** Run `tools/wyscout_benchmark.py` on the Wyscout download and rerun this script. Until then the numbers below are the synthetic side only.

| metric | synthetic mean (sd) | Wyscout mean (sd) | gap in Wyscout sds |
| --- | --- | --- | --- |
| passes | 948.2 (94.4) | pending | pending |
| pass_accuracy | 0.810 (0.015) | pending | pending |
| shots | 23.8 (5.0) | pending | pending |
| goals | 2.7 (1.6) | pending | pending |
| fouls | 13.7 (2.6) | pending | pending |
| corners | 6.6 (2.6) | pending | pending |

Definitions differ slightly: Wyscout counts free kick shots and penalties under other event names, and has no event for a pass that was blocked. Treat gaps under about 1 sd as consistent.
