# Spatial variation in synthetic reconstructions

`scripts/analyse_spatial_variation.py` implements Table B.1 and the SchemaGAN output-summary comparison in Section 4.4.
It uses the four completed baseline methods, with TabPFN's predictive medians, Kriging's means, nearest-neighbour predictions and individually generated SchemaGAN reconstructions.

## Calculation

For each section, compute absolute differences between every pair of pixels separated vertically by 1, 2, 4 or 8 rows, and horizontally by 1, 2, 4, 8, 16, 32, 64 or 128 columns.
Use both the mean and median of these absolute differences, including pairs involving observed input pixels.
Do not divide by grid separation and do not identify layers or restrict pairs to within-layer locations.
Differences are calculated after conversion to float64.

Divide each reconstructed descriptor by the corresponding target descriptor in the same section.
For SchemaGAN, form these ratios for each of the 100 draws, then average them within each section.
Finally, report the median of each ratio over the 4,000 test sections.
The median over sections is taken after the within-section draw average, not before it.
A zero target descriptor makes the ratio undefined and is rejected; all target denominators were positive in the manuscript calculation.
Zero reconstructed median differences are retained, including those from nearest-neighbour interpolation.

For the output-summary comparison, instead calculate the descriptors of SchemaGAN's pixelwise median reconstruction, divide by the same section's target descriptors, and then take medians across sections.
Compare these with the medians from the individually scored draws.
The resulting change in a reported median is distinct from the maximum change in an individual section's ratio; `findings.json` reports both separately.

These quantities describe spatial variation of grid values.
They do not directly evaluate boundary locations, layer connectivity or engineering usefulness.

## Manuscript reference values

For mean absolute differences at a vertical separation of one row, the medians of the ratios were 0.65 for TabPFN and 0.76 for SchemaGAN.
Across horizontal separations, they ranged from 0.37 to 0.82 for TabPFN and from 0.74 to 1.23 for SchemaGAN.
The maximum absolute changes in the reported medians when using SchemaGAN's pixelwise median were 0.0216256101 for mean absolute differences and 0.0238035833 for median absolute differences, both below 0.03.
These are references from the full manuscript runs, not expected results for a prefix or a reduced set of draws.
New predictions can differ with GPU hardware and numerical libraries, as described in the experiment guide.

## Provenance

The scientific calculation was ported from the manuscript source repository at commit `22322a44e6b37dd2323be34450b15f2c30206416`.
The source files at that commit were:

| Source file | SHA-256 |
|---|---|
| `scripts/analyze_spatial_difference_ratios.py` | `c12dbefe7aa1e79e50fd3ab5889f799191102137a702dde8fa6530dee1a735ca` |
| `scripts/analyze_spatial_output_sensitivity.py` | `9370f8803f1aabfea9dddc76227c409d358662b65439e2daca7f7389adecc631` |

The public entry point reads the portable run directories produced by this repository instead of the manuscript's local result paths.
It uses the existing pinned analysis environment and processes predictions sequentially with bounded section blocks.
The calculation and aggregation rules above are unchanged.

## Validation

The port was checked against the original saved results at section indices 0, 777, 2000 and 3999, using all four methods and all 100 SchemaGAN draws.
Target descriptors, per-section ratios and the pixelwise-median ratios matched exactly at those indices.
Aggregation of the previously verified 4,000-section scores reproduced all 96 table values and all 24 output-summary comparisons.
The command also completed with this repository's two-section, two-draw demo runs.
Unit tests cover independently enumerated pixel pairs, aggregation order, zero medians and invalid or mismatched cached inputs.
The full inference campaign and full descriptor calculation were not repeated for this port.
