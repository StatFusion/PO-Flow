# Final-result provenance

`final_results.csv` is the compact release record linking every canonical
PO-Flow config to its final five-seed aggregate. Scalar rows come from the
safety-selected scalar IPW result set. Vector/image rows come from the final
optimization-v3 set; `retained_architecture_v2` means that the v3 challenger
failed the safety gate and the frozen v2 configuration was retained.

All means and sample standard deviations use seeds 42--46. The distributional
columns are W1/W1-mu for scalar outcomes and sliced-W1/sliced-W1-mu for
vector/image outcomes. Vector outcomes are evaluated in training-factual
standardized coordinates; MorphoMNIST pixels are scaled to [-1,1].

Raw logs, checkpoints, development grids, and rejected candidates remain
outside Git. The release verifier instead checks the exact frozen hashes of the
core code, final configs, and processed data used by these rows.
