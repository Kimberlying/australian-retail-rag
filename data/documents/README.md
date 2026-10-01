# Documents

Put permitted public `.pdf`, `.md`, or `.txt` documents here and rerun:

```bash
retail-rag ingest
```

## Metadata

Retrieval can filter by `company` and `fiscal_year`, so every document should say what it covers:

- **Markdown / text:** a front matter block at the top of the file.

  ```text
  ---
  company: Coles
  fiscal_year: 2025
  source_type: public
  ---
  ```

- **PDF:** a sidecar file next to it, named `<file>.pdf.meta.json`, for example `{"company": "Coles", "fiscal_year": [2025, 2024]}`. List every fiscal year the document reports on, including prior-year comparatives and outlook years.

Leave `fiscal_year` out for timeless documents such as policies: they stay eligible for any fiscal-year question.

## What is here

- `coles_fy25_public_snapshot.md`: a small public-data snapshot. Refresh its facts from the official source URL before using the project in a public portfolio.
- `synthetic_*.md`: clearly labelled synthetic policies for the fictional retailer Harbourline Retail.
- `harbourline_fy25_annual_report.pdf`: a synthetic annual report, generated reproducibly by `scripts/make_sample_report.py`.
