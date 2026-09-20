Project 1 — Images of the Russian Empire
Sohum Phadke — CS 180

HOW TO RUN
-----------
1. Install dependencies:
     pip install numpy scipy scikit-image

2. Place the provided Prokudin-Gorskii plates (cathedral.jpg,
   monastery.jpg, tobolsk.jpg, church.tif, emir.tif, harvesters.tif,
   icon.tif, melons.tif, self_portrait.tif, siren.tif,
   three_generations.tif, ilemselga.tif, religous_painting.tif,
   wharf.tif) plus my 3 own selections
   (master-pnp-prok-00000-00026a.tif.tiff,
   master-pnp-prok-00400-00403a.tif.tiff,
   master-pnp-prok-00400-00448a.tif.tiff) into a sibling folder
   named "data/" next to this code/ folder, i.e.:

       project1/
         code/        <- this folder (main.py, align.py, ...)
         data/        <- put the .jpg / .tif plates here
         outputs/     <- created automatically

3. From inside code/, run:
     python main.py

   This reproduces every result on the webpage: it runs single-scale
   alignment on the 3 small JPEGs and pyramid alignment on all 17
   images, under both the NCC and L2 metrics, and writes:
     - outputs/<name>_<metric>.jpg   (colorized output images)
     - outputs/results.md            (offset + timing table)
     - outputs/results.json          (same data, machine-readable)

4. (Optional) To reproduce the emir.tif feature-strategy comparison
   shown in "The Journey" section of the webpage, run:
     python explore_features.py emir.tif --metric ncc
   This tries identity / percentile-stretch / histogram-equalization
   / Sobel-gradient features and saves each result to
   outputs/exploration/.

FILE OVERVIEW
-------------
main.py             - entry point; calls experiments.main()
align.py            - core implementation: channel splitting, the
                       single-scale exhaustive search, the recursive
                       coarse-to-fine pyramid, the L2/NCC metrics,
                       and the feature transforms (identity,
                       percentile stretch, histogram equalization,
                       Sobel gradient magnitude)
experiments.py       - batch driver: runs align.py over every file
                       in data/, for both metrics, logging offsets
                       and timing to outputs/results.md / .json
explore_features.py  - standalone script for comparing feature
                       strategies on a single image (used to debug
                       and fix emir.tif's misalignment)

NOTES
-----
- Single-scale and pyramid alignment, and all feature/metric code,
  are implemented from scratch (numpy/scipy only used for basic
  array ops, Gaussian blur for downsampling, and Sobel filtering).
- Expect ~15-70s per large .tif under the pyramid path, ~1s per
  small .jpg under single-scale search, on a normal laptop CPU.
