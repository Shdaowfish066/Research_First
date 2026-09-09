"""
=============================================================================
CAMERA-READY COMPLIANCE CHECK  (iCONEECT 2026)
=============================================================================
Checks the compiled PDF against every hard requirement on the conference
guidelines page, plus the usual LaTeX failure modes. Exits non-zero if any
blocking check fails, so it can gate a build.

    python verify_camera_ready.py [paper.pdf]

Checks
  1. page count <= 6                      (guidelines: "cannot exceed 6")
  2. page size is A4 or US Letter          (guidelines)
  3. file size < 5 MB                      (guidelines)
  4. every font embedded                   (guidelines)
  5. no Type 3 fonts                       (IEEE PDF eXpress rejects these)
  6. no unresolved references (??)         (LaTeX)
  7. no leftover placeholders              ([AUTHOR 1 NAME], TBD, Anonymous)
  8. header and footer strings present on page 1, and page 1 only
  9. no page numbers                       (IEEEtran conference default)
=============================================================================
"""

import os
import re
import sys
import subprocess
import zlib

PDF = sys.argv[1] if len(sys.argv) > 1 else "Iconeect2026_camera_ready.pdf"
TEX = "Iconeect2026_camera_ready.tex"
MAX_PAGES = 6
MAX_MB = 5.0
HEADER_KEY = "Next-Generation Electrical"
FOOTER_KEY = "979-8-3195-4810-8"

ok = True
def report(name, passed, detail="", blocking=True):
    global ok
    mark = "PASS" if passed else ("FAIL" if blocking else "WARN")
    print(f"  [{mark}] {name}" + (f"  --  {detail}" if detail else ""))
    if not passed and blocking:
        ok = False


def pages_and_boxes(raw):
    """Page count and MediaBoxes without external tools."""
    n = len(re.findall(rb"/Type\s*/Page[^s]", raw))
    boxes = [tuple(round(float(x)) for x in m.split())
             for m in re.findall(rb"/MediaBox\s*\[([^\]]*)\]", raw)]
    return n, boxes


def page_texts(raw):
    """Decompressed content streams, one per stream, best effort."""
    out = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.S):
        blob = m.group(1)
        try:
            out.append(zlib.decompress(blob).decode("latin-1", "ignore"))
        except Exception:
            out.append(blob.decode("latin-1", "ignore"))
    return out


def pdf_strings(stream_text):
    """Text drawn by the page, extracted from PDF string literals."""
    return " ".join(re.findall(r"\((?:\\.|[^\\()])*\)", stream_text))


print(f"\nVerifying {PDF}\n" + "=" * 66)
if not os.path.exists(PDF):
    sys.exit(f"  [FAIL] {PDF} does not exist -- compile first.")
raw = open(PDF, "rb").read()

# 1-3 -----------------------------------------------------------------------
npages, boxes = pages_and_boxes(raw)
report(f"page count <= {MAX_PAGES}", 0 < npages <= MAX_PAGES, f"{npages} pages")

A4 = (595, 842)
LT = (612, 792)
sizes = {(round(b[2] - b[0]), round(b[3] - b[1])) for b in boxes} if boxes else set()
good = sizes and all(s in (A4, LT) for s in sizes)
report("page size A4 or US Letter", bool(good),
       ", ".join(f"{w}x{h}pt" for w, h in sorted(sizes)) or "unknown")

mb = os.path.getsize(PDF) / 1024 / 1024
report(f"file size < {MAX_MB} MB", mb < MAX_MB, f"{mb:.2f} MB")

# 4-5 -----------------------------------------------------------------------
fonts = subprocess.run(["pdffonts", PDF], capture_output=True, text=True)
if fonts.returncode == 0 and fonts.stdout.strip():
    lines = [l for l in fonts.stdout.splitlines()[2:] if l.strip()]
    not_emb = [l for l in lines if len(l.split()) > 3 and l.split()[-4] == "no"]
    type3 = [l for l in lines if "Type 3" in l]
    report("all fonts embedded", not not_emb,
           f"{len(lines)} fonts" + (f"; NOT embedded: {len(not_emb)}" if not_emb else ""))
    report("no Type 3 fonts", not type3, f"{len(type3)} found" if type3 else "")
else:
    emb = raw.count(b"FontFile2") + raw.count(b"FontFile3") + raw.count(b"FontFile")
    report("all fonts embedded", emb > 0,
           f"pdffonts unavailable; {emb} embedded font files found", blocking=False)
    report("no Type 3 fonts", b"/Type3" not in raw,
           "pdffonts unavailable; scanned raw PDF", blocking=False)

# 6-7 -----------------------------------------------------------------------
# Rendered text, not raw streams: embedded font binaries contain literal "??"
# byte sequences and produce false positives if the streams are scanned.
try:
    import pymupdf
    doc = pymupdf.open(PDF)
    pagetext = [p.get_text() for p in doc]
except Exception:
    pagetext = [pdf_strings(s) for s in page_texts(raw)]
alltext = "\n".join(pagetext)

bad = [(i, t[max(0, m.start() - 50):m.start() + 15])
       for i, t in enumerate(pagetext, 1)
       for m in re.finditer(r"\?\?", t)]
report("no unresolved references (??)", not bad,
       f"{len(bad)} on page(s) {sorted({i for i,_ in bad})}" if bad else "")

if os.path.exists(TEX):
    tex = open(TEX, encoding="utf-8").read()
    left = re.findall(r"\[(?:AUTHOR|DEPT|INSTITUTION|CITY|COUNTRY|EMAIL)[^\]]*\]", tex)
    left += re.findall(r"\bTBD\b", tex)
    left += ["Anonymous"] if "Anonymous" in tex else []
    report("no leftover placeholders in .tex", not left,
           f"{len(left)} found: {sorted(set(left))[:6]}" if left else "", blocking=False)

# 8 -------------------------------------------------------------------------
def pages_containing(key):
    k = key.replace(" ", "").lower()
    return [i for i, t in enumerate(pagetext, 1)
            if k in t.replace(" ", "").replace("\n", "").lower()]

hdr = pages_containing(HEADER_KEY)
ftr = pages_containing(FOOTER_KEY)
report("first-page header present", hdr == [1], f"pages {hdr}")
report("first-page footer present", ftr == [1], f"pages {ftr}")


# IEEEtran conference prints no folio; a bare number alone on the last line
# of a page would indicate one.
numbered = [i for i, t in enumerate(pagetext, 1)
            if re.search(r"\n\s*\d{1,2}\s*$", t)]
report("no page numbers", not numbered, f"pages {numbered}" if numbered else "",
       blocking=False)


print("\n" + "=" * 66)
print("Manual checks still required:")
print("  - render each page and look at it (no clipped tables, no orphan captions)")
print("  - similarity report below 30%")
print("  - author block filled in, unused blocks deleted")
print("  - IEEE PDF eXpress conversion with conference ID 71331X")
print("=" * 66)
print("RESULT:", "all blocking checks passed" if ok else "BLOCKING FAILURES ABOVE")
sys.exit(0 if ok else 1)
