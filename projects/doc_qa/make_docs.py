"""
Generate the sample document set in docs/: Markdown policies, release notes and a PDF manual
for a fictional company, Northwind Labs. The content is designed to exercise retrieval:
exact codes (E204), paraphrases ("time off" vs "PTO accrual"), numbers, and a fact that
changed between firmware versions.

The PDF is written with a tiny built-in writer (standard Helvetica, one text block per
page), so no PDF library is needed to create it; pypdf reads it back during ingestion.
Run from the repo root:  python -m projects.doc_qa.make_docs
"""
from pathlib import Path

DOCS = Path(__file__).parent / "docs"

HANDBOOK = """# Northwind Labs Employee Handbook

## Paid time off

### Accrual
Full-time employees accrue 1.5 days of paid time off (PTO) per month, which is 18 days per year.
Part-time employees accrue PTO in proportion to their scheduled hours.

### Carryover
Up to 5 unused PTO days carry over into the next calendar year. Days above that limit expire on
January 31. Unused PTO is not paid out except where local law requires it.

### Requesting time off
Request PTO in the HR portal at least 2 weeks in advance for absences longer than 3 days.
Managers respond within 3 business days.

## Parental leave
Primary caregivers receive 16 weeks of fully paid parental leave. Secondary caregivers receive
6 weeks. Leave can start up to 2 weeks before the expected birth or adoption date and must be
taken within 12 months.

## Remote work
Employees may work remotely up to 3 days per week with manager approval. Fully remote roles are
listed as such in the job description. The company provides a one-time home office stipend of
$500 for new employees.

## Expenses

### Meals and travel
Meal expenses are reimbursed up to $75 per day while travelling. Economy class is required for
flights under 6 hours; premium economy is allowed for longer flights.

### Receipts and deadlines
Receipts are required for any expense over $25. Submit expense reports within 30 days of the
expense; reports submitted later need written approval from a director.

## Equipment
Every employee receives a laptop that is refreshed every 3 years. Lost or stolen equipment must
be reported to IT within 24 hours.
"""

SECURITY = """# Northwind Labs IT Security Policy

## Passwords and multi-factor authentication
Passwords must be at least 14 characters long. Multi-factor authentication (MFA) is mandatory for
email, the VPN and all production systems. Passwords do not expire on a schedule, but must be
changed immediately if a compromise is suspected.

## Reporting security incidents
Report suspected security incidents, including phishing emails, to security@northwind.example
within 1 hour of discovery. Do not forward suspicious emails to colleagues; use the Report Phish
button instead.

## Data classification
Northwind data has four levels: Public, Internal, Confidential and Restricted. Restricted data,
such as customer payment details, must never be stored on laptops or personal devices.

## Devices and VPN
Laptops must use full-disk encryption and lock after 5 minutes of inactivity. The VPN is required
when accessing internal systems from outside the office network.
"""

RELEASE_NOTES = """# X200 Router Firmware Release Notes

## Version 3.2 (September 2026)
- The admin console moved from http://192.168.1.1 to http://router.local. The old address
  redirects until firmware 4.0.
- Added WPA3-only mode under Wireless > Security.
- Fixed an issue where error E204 was shown during brief ISP outages.

## Version 3.1 (May 2026)
- Added scheduled automatic updates (default: Sundays at 3:00 am).
- Improved 6 GHz band stability.
- Guest network bandwidth can now be limited per device.
"""

MANUAL_PAGES = [
    ["X200 Router User Manual", "", "Page 1 - Getting started",
     "Connect the WAN port to your modem with the included cable.",
     "Power on the router and wait for the status LED to turn solid white (about 2 minutes).",
     "Open the admin console at http://192.168.1.1 and sign in with the password",
     "printed on the label under the router.",
     "Firmware 3.2 and later use http://router.local instead (see the release notes)."],
    ["Page 2 - Status LED", "",
     "Solid white: connected to the internet.",
     "Pulsing white: starting up or updating firmware. Do not unplug.",
     "Solid amber: no internet connection. Check the modem and the WAN cable.",
     "Blinking red: an error. Read the code in the admin console (see page 3)."],
    ["Page 3 - Error codes", "",
     "E101 - Overheating. Move the router to a ventilated place; it restarts when cool.",
     "E204 - WAN link down. The modem is not responding. Restart the modem, then the router.",
     "E310 - Firmware verification failed. Download the firmware again and retry the update.",
     "E415 - Too many devices. The X200 supports up to 128 connected devices."],
    ["Page 4 - Reset and firmware", "",
     "Restart: hold the reset button for 3 seconds.",
     "Factory reset: hold the reset button for 15 seconds until the LED blinks amber.",
     "A factory reset erases all settings, including Wi-Fi names and passwords.",
     "Firmware updates are installed from Settings > Firmware > Check for updates."],
    ["Page 5 - Specifications and warranty", "",
     "Wi-Fi 6E tri-band (2.4, 5 and 6 GHz), up to 5.4 Gbps combined.",
     "Ports: one 2.5 Gbps WAN port, four 1 Gbps LAN ports, one USB-C port.",
     "Coverage: up to 2,500 square feet.",
     "Warranty: 2 years limited hardware warranty from the date of purchase."],
]


def write_pdf(path: Path, pages):
    """Minimal PDF 1.4 writer: one Helvetica text block per page (ASCII text only)."""
    def esc(s):
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects = []  # PDF objects, 1-indexed in the file
    n_pages = len(pages)
    page_ids = [3 + 2 * i for i in range(n_pages)]
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {n_pages} >>")
    font_id = 3 + 2 * n_pages
    for lines in pages:
        text = "BT /F1 11 Tf 14 TL 60 760 Td " + " ".join(f"({esc(line)}) Tj T*" for line in lines) + " ET"
        content_id = len(objects) + 2
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>")
        objects.append(f"<< /Length {len(text)} >>\nstream\n{text}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")

    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def main():
    DOCS.mkdir(exist_ok=True)
    (DOCS / "employee_handbook.md").write_text(HANDBOOK, encoding="utf-8")
    (DOCS / "it_security_policy.md").write_text(SECURITY, encoding="utf-8")
    (DOCS / "x200_release_notes.md").write_text(RELEASE_NOTES, encoding="utf-8")
    write_pdf(DOCS / "x200_router_manual.pdf", MANUAL_PAGES)
    print(f"Wrote {len(list(DOCS.iterdir()))} documents to {DOCS}")


if __name__ == "__main__":
    main()
