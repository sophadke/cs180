# Project 2: Fun with Filters and Frequencies

## Folder layout

```
project 3/
  funfilter.py        your original Part 1.1 script (left as-is)
  code/               all the project code
    convolution.py      1.1 - your numpy convolutions (4 loops, 2 loops)
    part1_filters.py    1.1-1.3 + bells & whistles (orientation)
    part2_sharpen.py    2.1 unsharp mask
    part2_hybrid.py     2.2 hybrid images (+ color bells & whistles)
    part2_blend.py      2.3 stacks + 2.4 multiresolution blending
    pick_points.py      click alignment points for a hybrid pair
    utils.py            loading/saving, Gaussian, blur, FFT plots
    run_all.py          runs everything
  images/             INPUT images go here
  web/
    index.html          the project website
    style.css
    results/            OUTPUT images (made by run_all.py)
```

## Images to put in `images/`

| File | Used for |
|---|---|
| `me.jpg` | 1.1 - a photo of yourself (any color photo; read as grayscale) |
| `cameraman.png` | 1.2/1.3 - optional; skimage's built-in copy is used if missing |
| `taj.jpg` | 2.1 - from the assignment page |
| `sharpen_mine.jpg` | 2.1 - a blurry photo of your choice |
| `sharp_test.jpg` | 2.1 - a sharp photo (it gets blurred then re-sharpened) |
| `DerekPicture.jpg`, `nutmeg.jpg` | 2.2 - from the starter code zip |
| `pair2_far.jpg`, `pair2_close.jpg` | 2.2 - your hybrid #2 (far = seen from a distance) |
| `pair3_far.jpg`, `pair3_close.jpg` | 2.2 - your hybrid #3 |
| `apple.jpeg`, `orange.jpeg` | 2.3/2.4 - from spline.zip |
| `blend2_a.jpg`, `blend2_b.jpg` | 2.4 - your straight-seam blend (A on top) |
| `blend3_a.jpg`, `blend3_b.jpg` | 2.4 - your irregular-mask blend (A inside the ellipse) |

Different filenames? Change the lists at the top of each `part*.py` file.
For the irregular mask you can also draw your own black/white PNG (white = image A),
save it in `images/`, and put its filename in the `BLENDS` list in `part2_blend.py`.

## Run

```bash
cd code
# once per hybrid pair: click the two eyes on each image
python3 pick_points.py DerekPicture.jpg nutmeg.jpg derek_nutmeg
python3 pick_points.py pair2_far.jpg pair2_close.jpg pair2
python3 pick_points.py pair3_far.jpg pair3_close.jpg pair3

python3 run_all.py
```

Then open `web/index.html`. The 1.1 timing table fills in automatically.
Search the page for the red `[...]` notes; those need your own words.
