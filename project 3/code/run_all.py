import json
import os
import re

import part1_filters
import part2_blend
import part2_hybrid
import part2_sharpen
from utils import OUT_DIR, ROOT


def fill_timings():
    """Write the 1.1 timings into index.html between the TIMINGS markers."""
    path = os.path.join(OUT_DIR, "timings.json")
    page = os.path.join(ROOT, "web", "index.html")
    if not (os.path.exists(path) and os.path.exists(page)):
        return
    with open(path) as f:
        t = json.load(f)
    rows = "".join(
        f"<tr><td>{k}</td><td>{v['four_loops']:.3f} s</td><td>{v['two_loops']:.3f} s</td>"
        f"<td>{v['scipy'] * 1000:.2f} ms</td><td>{'yes' if v['matches_scipy'] else 'NO'}</td></tr>"
        for k, v in t.items() if isinstance(v, dict))
    table = (f"<!--TIMINGS--><table><tr><th>kernel</th><th>4 loops</th><th>2 loops</th>"
             f"<th>scipy</th><th>matches scipy</th></tr>{rows}</table>"
             f"<p class=\"note\">Image size: {t['image_size'][0]} &times; {t['image_size'][1]} pixels.</p><!--/TIMINGS-->")
    with open(page) as f:
        html = f.read()
    html = re.sub(r"<!--TIMINGS-->.*?<!--/TIMINGS-->", table, html, flags=re.S)
    with open(page, "w") as f:
        f.write(html)
    print("Filled timing table in web/index.html")


if __name__ == "__main__":
    part1_filters.run()
    part2_sharpen.run()
    part2_hybrid.run()
    part2_blend.run()
    fill_timings()
    print("Done. Open web/index.html to see the results.")
