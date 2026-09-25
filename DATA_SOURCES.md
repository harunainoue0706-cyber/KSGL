# Benchmark data sources

The core repository does not duplicate the full external graph benchmark collections. `scripts/setup_data.py` retrieves the public source files recorded in `configs/benchmark_registry.json`, converts formats when needed, validates graph counts and strongly-regular parameters, and writes a local manifest.

Public source endpoints used by the bootstrap include:

- https://users.cecs.anu.edu.au/~bdm/data/graphs.html
- https://www.maths.gla.ac.uk/~es/srgraphs.php
- https://github.com/GraphPKU/BREC

The compact `SRG(25,12,5,6)` graph6 collection in `validation/` is included for a self-contained verification run.

Before republishing external raw datasets, verify the redistribution and citation requirements of the corresponding upstream source.
