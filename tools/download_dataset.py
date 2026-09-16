"""
Member 6 (Data & DevOps) — real image dataset downloader.

WHY THIS SCRIPT EXISTS
-----------------------
The images below are genuine, publicly released satellite imagery:
  - 3 official ISRO Chandrayaan-2 gallery frames (isro.gov.in)
  - 4 NASA Lunar Reconnaissance Orbiter Camera (LROC) images, each with a
    real, published crater/rille name and coordinates (lroc.asu.edu)

They are NOT downloadable from inside this build environment (no internet
here), so run this script yourself, on any machine that has internet:

    cd backend
    python download_dataset.py

It saves files into backend/images/dataset/ using the exact filenames
already referenced in data/metadata.json, so embed_images.py and main.py
pick them up with zero extra configuration.

WHAT IT DOES NOT DO
--------------------
It does not fabricate data: if a download fails (site restructured, image
moved, network hiccup), it tells you exactly which file failed and why,
instead of silently leaving a gap. Re-run the script after checking a
failed URL manually in a browser — sites like lroc.asu.edu occasionally
reorganize their news archive.

If you get real Chandrayaan-2 imagery from PRADAN later, just add more
entries here (or drop files straight into backend/images/dataset/ and add
matching entries to data/metadata.json) — nothing else needs to change.
"""
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

OUT_DIR = Path(__file__).parent / "images" / "dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (SatQuery dataset fetcher; contact: team)"}
TIMEOUT_SECONDS = 30

# Each entry is one of:
#   {"filename": ..., "direct_url": ...}          -> downloaded as-is
#   {"filename": ..., "source_page": ...}         -> the page's og:image
#                                                     (or first <img>) is
#                                                     resolved and downloaded
DATASET = [
    {
        "filename": "chandrayaan2_vikram_descent_1.png",
        "direct_url": "https://www.isro.gov.in/media_isro/image/GSLV/slide1.png.webp",
        "credit": "ISRO — official Chandrayaan-2 gallery",
    },
    {
        "filename": "chandrayaan2_vikram_descent_2.png",
        "direct_url": "https://www.isro.gov.in/media_isro/image/GSLV/slide2.png.webp",
        "credit": "ISRO — official Chandrayaan-2 gallery",
    },
    {
        "filename": "chandrayaan2_vikram_descent_3.png",
        "direct_url": "https://www.isro.gov.in/media_isro/image/GSLV/presentationmoonphoto.png.webp",
        "credit": "ISRO — official Chandrayaan-2 gallery",
    },
    {
        "filename": "copernicus_crater.png",
        "direct_url": "https://www.lroc.asu.edu/news/uploads/LROCiotw/copern_erat_bw400mann.png",
        "credit": "NASA/GSFC/Arizona State University (LROC)",
    },
    {
        "filename": "aristarchus_crater.jpg",
        "source_page": "https://www.lroc.asu.edu/images/426",
        "credit": "NASA/GSFC/Arizona State University (LROC) — 'Aristarchus Spectacular!'",
    },
    {
        "filename": "tycho_crater.jpg",
        "source_page": "https://lroc.sese.asu.edu/posts/514",
        "credit": "NASA/GSFC/Arizona State University (LROC) — 'View From The Other Side'",
    },
    {
        "filename": "rima_marius_rille.jpg",
        "source_page": "https://www.lroc.asu.edu/images/108",
        "credit": "NASA/GSFC/Arizona State University (LROC) — sinuous rille imagery",
    },
]


def fetch_bytes(url: str) -> bytes:
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        return resp.read()


def extract_image_url(html: bytes) -> str | None:
    text = html.decode("utf-8", errors="ignore")
    m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', text, re.I)
    if m:
        return m.group(1)
    m = re.search(r'<img[^>]+src=["\']([^"\']+\.(?:jpg|jpeg|png))["\']', text, re.I)
    return m.group(1) if m else None


def main():
    ok, failed = [], []
    for item in DATASET:
        out_path = OUT_DIR / item["filename"]
        try:
            if "direct_url" in item:
                print(f"-> downloading {item['filename']} directly ...")
                data = fetch_bytes(item["direct_url"])
            else:
                print(f"-> resolving image for {item['filename']} from {item['source_page']} ...")
                html = fetch_bytes(item["source_page"])
                img_url = extract_image_url(html)
                if not img_url:
                    raise RuntimeError("could not find an og:image or <img> tag on the source page")
                img_url = urljoin(item["source_page"], img_url)
                print(f"   found image: {img_url}")
                data = fetch_bytes(img_url)

            if len(data) < 500:
                raise RuntimeError(f"downloaded file is suspiciously small ({len(data)} bytes) — likely an error page, not an image")

            out_path.write_bytes(data)
            print(f"   OK saved {out_path} ({len(data) / 1024:.1f} KB) - {item['credit']}")
            ok.append(item["filename"])
        except (URLError, HTTPError, RuntimeError, TimeoutError, OSError) as e:
            print(f"   FAILED: {e}")
            failed.append((item["filename"], str(e)))

    print(f"\nDone: {len(ok)} succeeded, {len(failed)} failed.")
    if failed:
        print("\nThese need manual attention — open the URL in a browser, save the image,")
        print(f"and drop it into {OUT_DIR} with the exact filename shown:")
        for filename, err in failed:
            print(f"  - {filename}: {err}")
        sys.exit(1)

    print("\nNext step: python embed_images.py")


if __name__ == "__main__":
    main()
