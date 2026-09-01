# Modifications from upstream MiniOneRec

This repository is a derivative of [AkaliKong/MiniOneRec](https://github.com/AkaliKong/MiniOneRec). The public snapshot differs from upstream in the following material ways:

- adapts SFT and GRPO execution to a Qwen2.5-1.5B single-GPU workflow;
- adds resumable RL checkpoint handling and evaluation snapshots;
- adds constrained decoding compatibility changes;
- adds FAISS, constrained, Sinkhorn, dead-code reset, CVQ-style, and RQ-KMeans++ SID experiments;
- adds CRID-inspired popularity-ranked SID construction;
- adds category-consistency evaluation and compact experiment reports;
- removes local absolute paths, large checkpoints, generated predictions, logs, and data from the public snapshot;
- replaces machine-specific launch commands with parameterized release scripts.

The upstream and derivative work are distributed under the Apache License, Version 2.0. See `LICENSE` and `NOTICE`.
