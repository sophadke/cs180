"""
main.py — entry point for reproducing Project 1 results.

Running this script:
  1. Reads every image in ../data/ (the 14 provided Prokudin-Gorskii
     plates + 3 of my own, downloaded from the LoC collection).
  2. Runs single-scale alignment (cathedral.jpg, monastery.jpg,
     tobolsk.jpg) and pyramid alignment (all files) under both the
     NCC and L2 metrics, using align.py.
  3. Saves a colorized .jpg per (file, metric) to ../outputs/, plus
     a full results table to ../outputs/results.md and
     ../outputs/results.json.

See README.txt for setup / how to run.
"""

from experiments import main

if __name__ == "__main__":
    main()
