# Data use

The benchmark ships task definitions, candidate indices and the frozen parsed cache, not
the source workbooks. Obtain each source under its original terms and verify the SHA-256
values below. The parsed cache is included so the reference evaluator can run without
redistributing the workbooks.

- rice: `data/processed/plant_counterinference/source_tables_v1/kitbase2026_supplement/Supplementary-Table-4.xlsx` (SHA-256 `4e8ca39186aad4f593e5b988d64f5c2ab38cbf2e730983c52eb8013d3d58ca54`; cache `24bcd3c1f6c218bcc9bcd1f87005b595c198b24ab69ece4fbede6ae44962b665`)
- alga: `data/processed/plant_counterinference/source_tables_v1/chlamy_conditions2022_pmc_supplement/41588_2022_1052_MOESM4_ESM.xlsx` (SHA-256 `ac78244c0501ac2b2b6455c3b389767d8d32ac6eb9a10fd7053eb51105d209ad`; cache `bd3dd479c3691c7bb7a131e28e48ce8c7633f250a88b51a65bcb3af170a03742`)

The frozen parsed cache used by this repository is at `data/processed/contrapositive_sampling_v1/`.
