"""Run all parts of Project 2 and update the timing table in the webpage."""

import json
import os
import re

import part1_filters
import part2_blend
import part2_hybrid
import part2_sharpen
from utils import OUT_DIR, ROOT


def fill_timings():
    timings_path = os.path.join(OUT_DIR, "timings.json")
    page_path = os.path.join(ROOT, "web", "index.html")

    if not os.path.exists(timings_path) or not os.path.exists(page_path):
        return

    with open(timings_path) as f:
        timings = json.load(f)

    rows = []
    for name, values in timings.items():
        if not isinstance(values, dict):
            continue

        matches = "yes" if values["matches_scipy"] else "NO"
        row = (
            f"<tr><td>{name}</td>"
            f"<td>{values['four_loops']:.3f} s</td>"
            f"<td>{values['two_loops']:.3f} s</td>"
            f"<td>{values['scipy'] * 1000:.2f} ms</td>"
            f"<td>{matches}</td></tr>"
        )
        rows.append(row)

    height, width = timings["image_size"]
    table = (
        "<!--TIMINGS-->"
        "<table><tr><th>kernel</th><th>4 loops</th><th>2 loops</th>"
        "<th>scipy</th><th>matches scipy</th></tr>"
        + "".join(rows)
        + "</table>"
        + f'<p class="note">Image size: {height} &times; {width} pixels.</p>'
        + "<!--/TIMINGS-->"
    )

    with open(page_path) as f:
        html = f.read()

    updated_html = re.sub(
        r"<!--TIMINGS-->.*?<!--/TIMINGS-->",
        table,
        html,
        flags=re.S,
    )

    with open(page_path, "w") as f:
        f.write(updated_html)

    print("Filled timing table in web/index.html")


if __name__ == "__main__":
    part1_filters.run()
    part2_sharpen.run()
    part2_hybrid.run()
    part2_blend.run()
    fill_timings()
    print("Done. Open web/index.html to see the results.")
