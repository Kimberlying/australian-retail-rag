"""Generate the synthetic Harbourline Retail FY25 annual report PDF.

The PDF exercises the page-aware ingestion path the way a real annual report
does: several pages, a running header and footer on every page (which ingestion
must strip), a financial table with prior-year comparatives, and metadata
(company, fiscal years) supplied through a ``.meta.json`` sidecar.

It is written with a ~60-line PDF writer below instead of a PDF library, so the
repository needs no extra dependency to rebuild it:

    uv run python scripts/make_sample_report.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "data" / "documents" / "harbourline_fy25_annual_report.pdf"
METADATA = {
    "company": "Harbourline Retail",
    # Every fiscal year the report states figures or plans for: FY25 results, FY24
    # comparatives, and the FY26 outlook.
    "fiscal_year": [2025, 2024, 2026],
    "source_type": "synthetic",
    "doc_type": "annual_report",
}

HEADER = "Harbourline Retail Limited | Annual Report 2025 | Synthetic portfolio data"

PAGES: list[list[str]] = [
    [
        "Harbourline Retail Limited",
        "Annual Report 2025",
        "For the financial year ended 30 June 2025 (FY25)",
        "",
        "Harbourline Retail is a fictional Australian supermarket retailer created for this "
        "portfolio project. Every figure in this report is synthetic and describes no real "
        "company.",
    ],
    [
        "Letter from the Chair and Chief Executive",
        "",
        "FY25 was a year of steady growth for Harbourline Retail. Sales revenue rose 5.6% to "
        "A$138.6 million, compared with A$131.2 million in FY24. Earnings before interest and tax "
        "(EBIT) increased to A$6.9 million from A$6.1 million, and net profit after tax (NPAT) was "
        "A$4.3 million, up from A$3.8 million.",
        "",
        "Online channels continued to grow. Click and collect and home delivery together accounted "
        "for 28.1% of orders in FY25, up from 24.7% in FY24. Click and collect orders met the "
        "2-hour picking standard 91% of the time.",
        "",
        "The Board has declared a fully franked final dividend of 6.5 cents per share, bringing "
        "the full-year dividend to 11.0 cents per share (FY24: 10.0 cents).",
    ],
    [
        "Financial summary (A$ million unless stated)",
        "",
        "Metric                       FY25     FY24",
        "Sales revenue                138.6    131.2",
        "Gross margin (%)             27.9     27.4",
        "EBIT                         6.9      6.1",
        "Net profit after tax         4.3      3.8",
        "Earnings per share (cents)   17.2     15.2",
        "Capital expenditure          8.1      7.4",
        "Net debt                     12.6     14.9",
        "",
        "Gross margin improved by 50 basis points, mainly from lower shrink and a better mix of "
        "fresh categories. Capital expenditure funded two store refurbishments and a new "
        "warehouse management system at the Sydney distribution centre.",
    ],
    [
        "Operations review",
        "",
        "Store network: Harbourline operated 12 stores at 30 June 2025, in New South Wales, "
        "Victoria, Queensland, South Australia, the Australian Capital Territory, and Tasmania. "
        "Braddon (ACT) opened in FY22 and remains the newest store.",
        "",
        "Loyalty: Harbourline Rewards reached 412,000 active members, up from 368,000 a year "
        "earlier. Members generated 57% of sales revenue.",
        "",
        "Inventory: the out-of-stock rate averaged 2.4% across the year, inside the 3% target set "
        "in the replenishment policy. Shrink was 1.2% of sales, down from 1.4% in FY24.",
        "",
        "People: Harbourline employed 1,640 team members at year end, 38% of them full time.",
    ],
    [
        "Outlook and risks",
        "",
        "In FY26 Harbourline plans to open two new stores in south-east Queensland and to extend "
        "same-day home delivery to all metropolitan stores. Planned capital expenditure is "
        "A$9.5 million.",
        "",
        "Key risks are cost-of-living pressure on customer spending, wage and energy cost "
        "inflation, supply chain disruption, and cyber security. The Board reviews each risk "
        "quarterly against the risk appetite statement.",
    ],
]


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _page_stream(lines: list[str], number: int, total: int) -> bytes:
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, 92) or [""])
    commands = ["BT", "/F1 9 Tf", "56 800 Td", f"({_escape(HEADER)}) Tj", "ET"]
    commands += ["BT", "/F1 11 Tf", "14 TL", "56 760 Td"]
    commands += [f"({_escape(line)}) Tj T*" for line in wrapped]
    commands += ["ET", "BT", "/F1 9 Tf", "280 40 Td", f"(Page {number} of {total}) Tj", "ET"]
    return "\n".join(commands).encode("latin-1")


def write_pdf(path: Path, pages: list[list[str]]) -> None:
    """Minimal PDF 1.4 writer: one Helvetica font, one content stream per page."""
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",  # page tree, filled in below
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    kids = []
    for number, lines in enumerate(pages, start=1):
        stream = _page_stream(lines, number, len(pages))
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        content_id = len(objects)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % content_id
        )
        kids.append(len(objects))
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % kid for kid in kids),
        len(kids),
    )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % index + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path.write_bytes(bytes(out))


def main() -> None:
    write_pdf(OUTPUT, PAGES)
    sidecar = OUTPUT.with_name(f"{OUTPUT.name}.meta.json")
    sidecar.write_text(json.dumps(METADATA, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} and {sidecar.name}")


if __name__ == "__main__":
    main()
