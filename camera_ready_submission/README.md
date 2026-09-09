# Camera-Ready Submission — iCONEECT 2026

Upload portal: https://cmt3.research.microsoft.com/iCONEECT2026
Deadline: 10 September 2026 (guidelines page; the site homepage shows 12
September, so treat 10 September as binding).

Replace `PID` in every filename with the real Paper ID before uploading.

---

## The 7 files CMT expects

Your paper is **conditionally accepted**, so `PID_Response.pdf` is mandatory
for you. That makes **5 mandatory files**, of which 2 are done here and 3
require your accounts and cannot be produced by anyone else.

| # | Filename | What it is | Required | Status |
|---|---|---|---|---|
| 1 | `PID.pdf` | Main camera-ready paper, output of PDF eXpress | Mandatory | **you** — see below |
| 2 | `PID_express.pdf` | IEEE PDF eXpress success / compliance message | Mandatory | **you** |
| 3 | `PID_copyright.pdf` | Signed IEEE electronic Copyright Form | Mandatory | **you** |
| 4 | `PID_Response.pdf` | Answers to reviewers | Mandatory (conditional accept) | **done** |
| 5 | `PID_payment.pdf` | Payment slip / registration confirmation | Mandatory | **you** |
| 6 | `PID_ieee.pdf` | Proof of active IEEE membership through Dec 2026 | If applicable | you, if claiming member rate |
| 7 | `PID_student.pdf` | Proof of student status | If applicable | you, if claiming student rate |

### Why only 2 files are in this folder

Files 2, 3 and 5 are produced by systems that require your login and your
signature. Nothing about them can be generated locally:

- **`PID_express.pdf`** is the confirmation IEEE PDF eXpress emails you after
  it checks the PDF. You have to run the check.
- **`PID_copyright.pdf`** is a legal signature on the IEEE eCF, signed inside
  CMT by an author.
- **`PID_payment.pdf`** is your registration receipt, which exists only after
  you pay.

What is in this folder now:

- `Iconeect2026_camera_ready.pdf` — the finished paper, ready to feed into
  PDF eXpress. It becomes `PID.pdf` after conversion.
- `PID_Response.pdf` — the reviewer response, ready to upload as-is.
- `source/` — LaTeX source, generated tables, figure PDFs.

---

## Step 1 — Fill in the author block

`source/Iconeect2026_camera_ready.tex` has placeholders for three authors:
`[AUTHOR n NAME]`, `[DEPT n]`, `[INSTITUTION n]`, `[CITY n]`, `[COUNTRY n]`,
`[EMAIL n]`. Delete any unused author block.

**Watch out:** LaTeX reads `\\` followed by `[` as `\\[length]`. That is why
the placeholders are written `\\{}` on the lines before `[CITY n]` and
`[EMAIL n]`. Keep the `{}` or the compile fails with
`Illegal unit of measure`.

Recompile and re-check the page count afterwards; the author block changes
the height of page 1.

```
pdflatex Iconeect2026_camera_ready.tex   # twice
python verify_camera_ready.py
```

## Step 2 — Similarity check

Must be **below 30%**. The Methodology and Literature Review are largely
carried over from the submitted version, so expect a match against your own
earlier submission; check how your tool treats that.

## Step 3 — IEEE PDF eXpress

**Conference ID: `71331X`**

1. Create a PDF eXpress account using that conference ID.
2. Upload the compiled PDF as a **PDF Check**.
3. Download the returned compliant file, rename it `PID.pdf`.
4. Save the success/compliance message as `PID_express.pdf`.

Both are mandatory. Keep the message, not just the converted PDF.

## Step 4 — Copyright form

Sign the IEEE eCF electronically inside CMT. Physical forms are not accepted.
Save the completed form as `PID_copyright.pdf`.

## Step 5 — Register and pay

At least one author must register and pay before the deadline. Save the
receipt or confirmation email as `PID_payment.pdf`. Add `PID_ieee.pdf` or
`PID_student.pdf` if you are claiming a member or student rate.

---

## Compliance status of the paper as built

| Requirement | Value |
|---|---|
| Pages (max 6) | 6 |
| Paper size | A4, 595×842 pt |
| File size (max 5 MB) | 0.31 MB |
| Fonts embedded | 21 of 21 |
| Type 3 fonts | none |
| Unresolved references | none |
| First-page header | present, page 1 only |
| First-page copyright footer | present, page 1 only |
| Page numbers | none |

## Rebuild

```
python make_paper_tables.py      # all tables, from the 2000-run CSVs
python make_figures.py           # both figures as vector PDF
pdflatex Iconeect2026_camera_ready.tex   # twice
python verify_camera_ready.py
```

Tables and figures are generated from data, never typed by hand. Any number
in the paper traces to a row in `analysis/` or in the four
`*_5_seed_variance_results/` folders.
