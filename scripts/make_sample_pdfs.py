"""Generate the sample corpus: a court judgment, a lease deed and exam notes.

The PDFs are laid out line-by-line with an explicit word-wrapper rather than a
reporting library, for two reasons:

* the output is byte-for-byte deterministic, so ``p.3 L14-16`` in the README is
  still ``p.3 L14-16`` on your machine;
* it keeps the repository free of extra dependencies — PyMuPDF already ships
  with the project.

All content is fictional. Names, case numbers, registration numbers and amounts
were invented for demonstration and resemble no real party or document.

Run::

    python -m scripts.make_sample_pdfs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw"

PAGE_W, PAGE_H = 595.0, 842.0  # A4 in points
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 64.0, 72.0, 68.0
BODY_SIZE, BODY_LEAD = 10.4, 15.6
FONT_BODY, FONT_BOLD, FONT_ITALIC = "tiro", "tibo", "tiit"


class PdfWriter:
    """Minimal flowing-text PDF writer with automatic pagination."""

    def __init__(self, title: str, *, font: str = FONT_BODY) -> None:
        self.doc = fitz.open()
        self.title = title
        self.font = font
        self.page: fitz.Page | None = None
        self.y = 0.0
        self._new_page()

    # ------------------------------------------------------------------ layout
    @property
    def text_width(self) -> float:
        return PAGE_W - 2 * MARGIN_X

    def _new_page(self) -> None:
        if self.page is not None:
            self._footer()
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = MARGIN_TOP

    def _footer(self) -> None:
        assert self.page is not None
        label = f"Page {self.page.number + 1}"
        width = fitz.get_text_length(label, fontname=FONT_ITALIC, fontsize=8.4)
        self.page.insert_text(
            fitz.Point(PAGE_W - MARGIN_X - width, PAGE_H - 40.0),
            label,
            fontname=FONT_ITALIC,
            fontsize=8.4,
            color=(0.42, 0.42, 0.46),
        )

    def _ensure(self, needed: float) -> None:
        if self.y + needed > PAGE_H - MARGIN_BOTTOM:
            self._new_page()

    def _write_line(self, text: str, *, font: str, size: float, indent: float = 0.0,
                    color: tuple[float, float, float] = (0.05, 0.05, 0.08)) -> None:
        self._ensure(BODY_LEAD)
        assert self.page is not None
        self.page.insert_text(
            fitz.Point(MARGIN_X + indent, self.y),
            text,
            fontname=font,
            fontsize=size,
            color=color,
        )
        self.y += BODY_LEAD

    def _wrap(self, text: str, *, font: str, size: float, width: float) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if fitz.get_text_length(candidate, fontname=font, fontsize=size) <= width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]

    # -------------------------------------------------------------- public API
    def spacer(self, amount: float = BODY_LEAD * 0.55) -> None:
        self.y += amount

    def title_block(self, lines: list[str]) -> None:
        for index, line in enumerate(lines):
            size = 13.6 if index == 0 else 11.0
            font = FONT_BOLD
            width = fitz.get_text_length(line, fontname=font, fontsize=size)
            self._ensure(BODY_LEAD)
            assert self.page is not None
            self.page.insert_text(
                fitz.Point((PAGE_W - width) / 2.0, self.y),
                line,
                fontname=font,
                fontsize=size,
                color=(0.04, 0.04, 0.10),
            )
            self.y += BODY_LEAD + (3.0 if index == 0 else 0.0)
        self.spacer()

    def heading(self, text: str) -> None:
        self._ensure(BODY_LEAD * 2.2)
        self.spacer(BODY_LEAD * 0.45)
        for line in self._wrap(text, font=FONT_BOLD, size=11.0, width=self.text_width):
            self._write_line(line, font=FONT_BOLD, size=11.0, color=(0.03, 0.10, 0.32))
        self.spacer(BODY_LEAD * 0.2)

    def paragraph(self, text: str, *, indent: float = 0.0) -> None:
        for line in self._wrap(text, font=self.font, size=BODY_SIZE, width=self.text_width - indent):
            self._write_line(line, font=self.font, size=BODY_SIZE, indent=indent)
        self.spacer(BODY_LEAD * 0.25)

    def clause(self, label: str, text: str) -> None:
        """A numbered clause: label on the first line, hanging indent after."""
        indent = 34.0
        wrapped = self._wrap(text, font=self.font, size=BODY_SIZE, width=self.text_width - indent)
        self._ensure(BODY_LEAD)
        assert self.page is not None
        self.page.insert_text(
            fitz.Point(MARGIN_X, self.y), label, fontname=FONT_BOLD, fontsize=BODY_SIZE, color=(0.05, 0.05, 0.08)
        )
        self.page.insert_text(
            fitz.Point(MARGIN_X + indent, self.y),
            wrapped[0],
            fontname=self.font,
            fontsize=BODY_SIZE,
            color=(0.05, 0.05, 0.08),
        )
        self.y += BODY_LEAD
        for line in wrapped[1:]:
            self._write_line(line, font=self.font, size=BODY_SIZE, indent=indent)
        self.spacer(BODY_LEAD * 0.25)

    def bullet(self, text: str) -> None:
        indent = 22.0
        wrapped = self._wrap(text, font=self.font, size=BODY_SIZE, width=self.text_width - indent)
        self._ensure(BODY_LEAD)
        assert self.page is not None
        self.page.insert_text(
            fitz.Point(MARGIN_X + 8.0, self.y), "\u2022", fontname=self.font, fontsize=BODY_SIZE
        )
        self.page.insert_text(
            fitz.Point(MARGIN_X + indent, self.y), wrapped[0], fontname=self.font, fontsize=BODY_SIZE
        )
        self.y += BODY_LEAD
        for line in wrapped[1:]:
            self._write_line(line, font=self.font, size=BODY_SIZE, indent=indent)

    def kv(self, key: str, value: str) -> None:
        """Aligned ``Key : value`` row, wrapped under the value column."""
        col = 158.0
        self._ensure(BODY_LEAD)
        assert self.page is not None
        self.page.insert_text(fitz.Point(MARGIN_X, self.y), key, fontname=FONT_BOLD, fontsize=BODY_SIZE)
        wrapped = self._wrap(value, font=self.font, size=BODY_SIZE, width=self.text_width - col)
        self.page.insert_text(fitz.Point(MARGIN_X + col, self.y), wrapped[0], fontname=self.font, fontsize=BODY_SIZE)
        self.y += BODY_LEAD
        for line in wrapped[1:]:
            self._write_line(line, font=self.font, size=BODY_SIZE, indent=col)

    def row(self, cells: list[str], widths: list[float], *, bold: bool = False) -> None:
        """A simple table row at fixed column offsets."""
        self._ensure(BODY_LEAD)
        assert self.page is not None
        font = FONT_BOLD if bold else self.font
        x = MARGIN_X
        for cell, width in zip(cells, widths):
            text = cell
            while fitz.get_text_length(text, fontname=font, fontsize=9.6) > width - 6 and len(text) > 4:
                text = text[:-2]
            self.page.insert_text(fitz.Point(x, self.y), text, fontname=font, fontsize=9.6)
            x += width
        self.y += BODY_LEAD

    def save(self, path: Path) -> Path:
        self._footer()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.set_metadata({"title": self.title, "author": "VeriRAG sample corpus"})
        self.doc.save(str(path), deflate=True)
        self.doc.close()
        return path


# ===========================================================================
# 1. Court judgment (property dispute)
# ===========================================================================
def build_case_law(out_dir: Path) -> Path:
    writer = PdfWriter("Sharma v. Metro Realty Developers Pvt. Ltd.")
    writer.title_block(
        [
            "IN THE HIGH COURT OF JUDICATURE AT BOMBAY",
            "CIVIL APPELLATE JURISDICTION",
            "FIRST APPEAL NO. 412 OF 2021",
        ]
    )
    writer.kv("Appellant", "Mr. Ramesh Sharma, aged 47 years, residing at 12/B Sundar Nivas, Vile Parle (East), Mumbai 400057")
    writer.kv("Respondent", "Metro Realty Developers Pvt. Ltd., CIN U70102MH2009PTC194522, registered office at Kalpataru Chambers, Andheri (East), Mumbai 400069")
    writer.kv("Coram", "Hon'ble Mr. Justice A. R. Deshpande")
    writer.kv("Reserved on", "27 February 2023")
    writer.kv("Pronounced on", "14 March 2023")
    writer.spacer()

    writer.heading("1. NATURE OF THE APPEAL")
    writer.paragraph(
        "This first appeal arises out of the judgment and order dated 8 September 2021 passed by the "
        "learned District Consumer Disputes Redressal Commission, Mumbai Suburban, in Consumer "
        "Complaint No. 233 of 2020, whereby the complaint filed by the present Appellant was dismissed "
        "on the sole ground that the delay in delivery of possession was attributable to force majeure."
    )
    writer.paragraph(
        "The Appellant assails that finding as perverse and contrary to the record. The Respondent "
        "supports the impugned order and contends that the Appellant is a speculative investor and not "
        "a consumer within the meaning of Section 2(7) of the Consumer Protection Act, 2019."
    )

    writer.heading("2. FACTS IN BRIEF")
    writer.clause(
        "2.1",
        "By an Agreement for Sale dated 9 June 2016, duly registered as Document No. BOM-4/7734/2016, "
        "the Respondent agreed to sell Flat No. B-1104 on the eleventh floor of Tower B of the project "
        "known as 'Metro Skyline', situated at Chakala, Andheri (East), Mumbai, admeasuring 1,185 square "
        "feet of carpet area, for a total consideration of Rs. 1,42,50,000 (Rupees One Crore Forty Two "
        "Lakh Fifty Thousand only), exclusive of stamp duty, registration charges and applicable taxes.",
    )
    writer.clause(
        "2.2",
        "Clause 9(a) of the said Agreement stipulated that possession of the said flat would be handed "
        "over on or before 31 December 2019, with a grace period of six months, that is, up to 30 June 2020.",
    )
    writer.clause(
        "2.3",
        "It is undisputed that the Appellant paid a sum of Rs. 1,28,25,000, being ninety per cent of the "
        "total consideration, in eleven instalments between 9 June 2016 and 22 August 2019. The payment "
        "particulars are set out in Exhibit C-4 and are not in dispute.",
    )
    writer.clause(
        "2.4",
        "The occupation certificate for Tower B was in fact granted by the Municipal Corporation of "
        "Greater Mumbai only on 19 November 2022, and possession was offered to the Appellant by letter "
        "dated 2 December 2022, that is, thirty-five months after the contractual date and twenty-nine "
        "months after expiry of the grace period.",
    )
    writer.clause(
        "2.5",
        "By notice dated 14 January 2020, the Appellant terminated the Agreement and called upon the "
        "Respondent to refund the amounts paid together with interest. The Respondent did not reply to "
        "the said notice, a fact conceded by its counsel during arguments.",
    )

    writer.heading("3. ISSUES FRAMED FOR DETERMINATION")
    writer.clause("(i)", "Whether the Appellant is a 'consumer' within the meaning of Section 2(7) of the Consumer Protection Act, 2019.")
    writer.clause("(ii)", "Whether the delay in delivery of possession is protected by the force majeure clause, being Clause 14 of the Agreement for Sale.")
    writer.clause("(iii)", "Whether the Appellant is entitled to refund with interest under Section 18 of the Real Estate (Regulation and Development) Act, 2016, and if so, at what rate.")
    writer.clause("(iv)", "What relief, if any, the Appellant is entitled to by way of compensation and costs.")

    writer.heading("4. SUBMISSIONS OF THE APPELLANT")
    writer.paragraph(
        "Mr. S. K. Iyer, learned counsel for the Appellant, submitted that the Appellant purchased the "
        "flat for the residential use of his family and that the Respondent has produced no material "
        "whatsoever to show any commercial intent. He relied upon the ratio in Newtech Promoters and "
        "Developers Pvt. Ltd. v. State of Uttar Pradesh, (2021) 5 SCC 1, to contend that the allottee's "
        "right to refund under Section 18(1) of the RERA Act is unconditional and does not depend upon "
        "proof of loss."
    )
    writer.paragraph(
        "He further submitted that the pandemic-related restrictions in Maharashtra operated for "
        "approximately seven months and cannot explain a delay of twenty-nine months, and that the "
        "Respondent's own annual report for the financial year 2018-19 attributed the delay to a "
        "shortfall in construction finance."
    )

    writer.heading("5. SUBMISSIONS OF THE RESPONDENT")
    writer.paragraph(
        "Ms. P. R. Kulkarni, learned counsel for the Respondent, urged that the Appellant already owns "
        "two residential flats in Mumbai and is therefore an investor. She relied upon Clause 14 of the "
        "Agreement, which lists 'epidemic, pandemic and any order of any statutory authority' among the "
        "events excusing performance, and submitted that the period from 24 March 2020 must be excluded."
    )
    writer.paragraph(
        "She alternatively submitted that if any interest is payable it ought to be restricted to the "
        "State Bank of India marginal cost of funds based lending rate, without any penal addition."
    )

    writer.heading("6. FINDINGS AND REASONS")
    writer.clause(
        "6.1",
        "On the first issue, mere ownership of other immovable property does not by itself convert an "
        "allottee into an investor. The burden of establishing a commercial purpose lies upon the party "
        "asserting it, and the Respondent has discharged no such burden. Following Imperia Structures "
        "Ltd. v. Anil Patni, (2020) 10 SCC 783, the Appellant is held to be a consumer. Issue (i) is "
        "answered in the affirmative.",
    )
    writer.clause(
        "6.2",
        "On the second issue, a force majeure clause must be construed strictly and cannot be invoked to "
        "excuse a delay that had already commenced before the supervening event. The contractual date of "
        "possession expired on 31 December 2019 and the grace period on 30 June 2020. The Respondent was "
        "therefore already in breach on 24 March 2020. At the highest, the Respondent may claim exclusion "
        "of seven months attributable to the pandemic, leaving an unexplained delay of twenty-two months. "
        "Issue (ii) is answered in the negative.",
    )
    writer.clause(
        "6.3",
        "On the third issue, Section 18(1) of the RERA Act confers upon the allottee an unqualified "
        "election either to withdraw from the project and claim refund with interest, or to continue and "
        "claim monthly interest for the period of delay. The Appellant validly exercised the first option "
        "by notice dated 14 January 2020. The rate of interest prescribed under Rule 18 of the "
        "Maharashtra Real Estate (Regulation and Development) Rules, 2017, being the State Bank of India "
        "highest marginal cost of lending rate plus two per cent, works out to 9.5 per cent per annum on "
        "the date of the notice.",
    )
    writer.clause(
        "6.4",
        "On the fourth issue, the Appellant has been kept out of both his money and his home for nearly "
        "three years and was compelled to reside in rented premises at a monthly rent of Rs. 62,000, as "
        "evidenced by Exhibit C-11. A separate award of compensation is therefore warranted, over and "
        "above interest, which compensates only for the deprivation of money.",
    )

    writer.heading("7. ORDER")
    writer.clause("7.1", "The appeal is allowed. The judgment and order dated 8 September 2021 of the District Commission is set aside.")
    writer.clause(
        "7.2",
        "The Respondent shall refund to the Appellant the sum of Rs. 1,28,25,000 together with simple "
        "interest at the rate of 9.5 per cent per annum from 1 January 2020 until the date of actual "
        "payment.",
    )
    writer.clause("7.3", "The Respondent shall pay to the Appellant compensation of Rs. 5,00,000 for mental agony and harassment.")
    writer.clause("7.4", "The Respondent shall pay the costs of this appeal quantified at Rs. 75,000.")
    writer.clause(
        "7.5",
        "The entire amount under paragraphs 7.2 to 7.4 shall be paid within ninety days from the date of "
        "this judgment. In default, the unpaid amount shall carry additional interest at the rate of 12 "
        "per cent per annum from the date of default until realisation.",
    )
    writer.spacer()
    writer.paragraph("Certified true copy. Registrar (Judicial), High Court of Judicature at Bombay.")

    return writer.save(out_dir / "case_sharma_v_metro_realty.pdf")


# ===========================================================================
# 2. Property lease deed
# ===========================================================================
def build_lease_deed(out_dir: Path) -> Path:
    writer = PdfWriter("Registered Lease Deed - Greenwood Residency C-704")
    writer.title_block(["REGISTERED LEASE DEED", "GREENWOOD RESIDENCY, FLAT NO. C-704, BENGALURU"])

    writer.paragraph(
        "THIS DEED OF LEASE is made and executed at Bengaluru on this the 5th day of April 2024, and "
        "registered before the Sub-Registrar, Bengaluru Urban (Whitefield), as Document No. "
        "BLR-3/4821/2024, Book 1, Volume 217, Pages 88 to 104."
    )

    writer.heading("BETWEEN THE PARTIES")
    writer.kv("Lessor", "Mrs. Kavita Menon, W/o Mr. Suresh Menon, aged 54 years, PAN AFZPM4471K, residing at 21 Palm Meadows, Ramagondanahalli, Bengaluru 560066")
    writer.kv("Lessee", "Mr. Arjun Rao, S/o Mr. Vikram Rao, aged 31 years, PAN BKQPR8823L, employed with Zentra Analytics India Pvt. Ltd., Embassy Tech Village, Bengaluru 560103")
    writer.kv("Witness 1", "Mr. Naveen Kumar, 44 Brookefield Layout, Bengaluru 560037")
    writer.kv("Witness 2", "Mrs. Latha Prabhu, 9 Sadaramangala Road, Bengaluru 560048")

    writer.heading("SCHEDULE A - DESCRIPTION OF THE DEMISED PREMISES")
    writer.paragraph(
        "Flat No. C-704 situated on the seventh floor of Block C of the residential apartment complex "
        "known as 'Greenwood Residency', bearing Municipal Khata No. 148/72, Survey No. 63/2 of "
        "Ramagondanahalli Village, Varthur Hobli, Whitefield, Bengaluru 560066, admeasuring 1,340 square "
        "feet of super built-up area and 985 square feet of carpet area, together with two covered car "
        "parking spaces bearing Nos. P-42 and P-43 in the basement level, and one storage unit bearing "
        "No. S-19."
    )
    writer.paragraph(
        "Boundaries of the said Block C are: on the East, the 40 feet wide Varthur Main Road; on the "
        "West, the common clubhouse and swimming pool; on the North, Block B of the same complex; and on "
        "the South, the property bearing Khata No. 148/73 belonging to Mr. Ganesh Iyer."
    )

    writer.heading("CLAUSE 1 - TERM AND LOCK-IN PERIOD")
    writer.clause("1.1", "The lease shall be for a term of thirty-three (33) months commencing from 1 May 2024 and expiring on 31 January 2027.")
    writer.clause(
        "1.2",
        "The first eleven (11) months of the term, that is up to 31 March 2025, shall constitute the "
        "lock-in period during which neither party shall be entitled to terminate this lease except for "
        "breach of a material covenant.",
    )
    writer.clause(
        "1.3",
        "Should the Lessee vacate the demised premises before expiry of the lock-in period, the Lessee "
        "shall be liable to pay the Lessor a sum equivalent to two (2) months' rent as liquidated "
        "damages, which the Lessor shall be entitled to adjust against the security deposit.",
    )

    writer.heading("CLAUSE 2 - RENT, ESCALATION AND MODE OF PAYMENT")
    writer.clause("2.1", "The monthly rent for the demised premises shall be Rs. 48,500 (Rupees Forty Eight Thousand Five Hundred only), payable in advance on or before the fifth (5th) day of every English calendar month.")
    writer.clause("2.2", "The rent shall stand escalated by six per cent (6%) upon completion of every eleven (11) months of the term, the first such escalation taking effect from 1 April 2025.")
    writer.clause("2.3", "All payments shall be made by electronic transfer to the Lessor's account No. 3874100019226 with Canara Bank, Whitefield Branch, IFSC CNRB0003874. Cash payments shall not be accepted.")
    writer.clause("2.4", "Any rent remaining unpaid beyond the tenth (10th) day of the month shall carry interest at eighteen per cent (18%) per annum calculated on a daily basis until payment.")
    writer.clause("2.5", "The Lessee shall deduct tax at source wherever required under Section 194-IB of the Income Tax Act, 1961, and shall furnish Form 16C to the Lessor within the prescribed time.")

    writer.heading("CLAUSE 3 - SECURITY DEPOSIT")
    writer.clause("3.1", "The Lessee has, on or before the execution hereof, paid to the Lessor an interest-free refundable security deposit of Rs. 2,91,000 (Rupees Two Lakh Ninety One Thousand only), being equivalent to six (6) months' rent, the receipt whereof the Lessor hereby acknowledges.")
    writer.clause("3.2", "The security deposit shall be refunded to the Lessee within twenty-one (21) days from the date of handing over vacant and peaceful possession of the demised premises, after adjusting arrears of rent, unpaid utility charges and the cost of repairing damage other than normal wear and tear.")
    writer.clause("3.3", "The security deposit shall not under any circumstances be adjusted against monthly rent during the currency of this lease.")

    writer.heading("CLAUSE 4 - MAINTENANCE, UTILITIES AND TAXES")
    writer.clause("4.1", "The Lessee shall pay the monthly apartment association maintenance charges of Rs. 3,750 directly to the Greenwood Residency Owners' Association.")
    writer.clause("4.2", "The Lessee shall bear all charges for electricity, water, piped gas, internet and direct-to-home television consumed in the demised premises, and shall produce paid receipts on demand.")
    writer.clause("4.3", "The Lessor shall bear the property tax, the corpus fund contribution and any special assessment levied by the Association for capital works.")
    writer.clause("4.4", "Minor repairs not exceeding Rs. 3,000 per instance shall be carried out by the Lessee at the Lessee's cost. Structural repairs, waterproofing, and replacement of sanitary and electrical fittings on account of normal wear shall be the responsibility of the Lessor.")

    writer.heading("CLAUSE 5 - USE OF THE DEMISED PREMISES")
    writer.clause("5.1", "The demised premises shall be used for residential purposes only and for no other purpose whatsoever. No commercial, industrial, religious or unlawful activity shall be permitted.")
    writer.clause("5.2", "The number of persons ordinarily residing in the demised premises shall not exceed five (5).")
    writer.clause("5.3", "The Lessee shall not sublet, assign, mortgage or part with possession of the demised premises or any part thereof without the prior written consent of the Lessor.")
    writer.clause("5.4", "The Lessee may keep not more than one (1) domestic pet, subject to compliance with the by-laws of the Association and prior written intimation to the Lessor.")
    writer.clause("5.5", "The Lessee shall not make any structural alteration, nor drill into load-bearing walls, nor install any external fixture visible from the facade, without prior written consent.")

    writer.heading("CLAUSE 6 - TERMINATION AND NOTICE")
    writer.clause("6.1", "After expiry of the lock-in period, either party may terminate this lease by giving the other party three (3) months' prior notice in writing, or by paying three months' rent in lieu of such notice.")
    writer.clause("6.2", "The Lessor may terminate this lease forthwith upon the Lessee committing default in payment of rent for two (2) consecutive months, or upon breach of Clause 5.1 or Clause 5.3.")
    writer.clause("6.3", "Upon termination or expiry, the Lessee shall hand over the demised premises in the same condition in which it was received, subject to normal wear and tear, along with all keys, access cards and the fixtures listed in Schedule B.")
    writer.clause("6.4", "The Lessee shall either repaint the interior of the demised premises before vacating or pay to the Lessor a lump sum of Rs. 22,000 towards repainting charges.")

    writer.heading("CLAUSE 7 - STAMP DUTY, REGISTRATION AND DISPUTES")
    writer.clause("7.1", "The stamp duty of Rs. 29,100 and the registration fee of Rs. 8,730 payable on this deed have been borne equally by both parties.")
    writer.clause("7.2", "Any dispute arising out of or in connection with this deed shall be referred to arbitration by a sole arbitrator appointed by mutual consent, under the Arbitration and Conciliation Act, 1996. The seat of arbitration shall be Bengaluru and the proceedings shall be conducted in the English language.")
    writer.clause("7.3", "Subject to Clause 7.2, the courts at Bengaluru alone shall have jurisdiction.")

    writer.heading("SCHEDULE B - INVENTORY OF FIXTURES AND FITTINGS")
    for item in [
        "Modular kitchen with granite counter, chimney and hob - 1 set",
        "Split air conditioners, 1.5 ton, in both bedrooms - 2 units",
        "Ceiling fans - 5 units",
        "Geysers, 25 litre, in both bathrooms - 2 units",
        "Wardrobes, full height, in master bedroom - 2 units",
        "Curtain rods with brackets - 7 units",
        "Modular light fittings and LED panels - 18 units",
        "Video door phone with intercom handset - 1 unit",
        "Access cards for lobby and clubhouse - 3 cards",
        "Keys: main door 3, bedroom doors 4, storage unit 1",
    ]:
        writer.bullet(item)

    writer.spacer()
    writer.paragraph(
        "IN WITNESS WHEREOF the parties hereto have set their hands to this deed on the day, month and "
        "year first above written, in the presence of the witnesses named above."
    )
    writer.kv("LESSOR", "Kavita Menon")
    writer.kv("LESSEE", "Arjun Rao")

    return writer.save(out_dir / "lease_deed_greenwood_c704.pdf")


# ===========================================================================
# 3. Study notes
# ===========================================================================
def build_study_notes(out_dir: Path) -> Path:
    writer = PdfWriter("DBMS Unit 3 - Normalization and Transactions", font="helv")
    writer.title_block(["DBMS UNIT 3 - EXAM REVISION NOTES", "NORMALIZATION, FUNCTIONAL DEPENDENCIES AND TRANSACTIONS"])
    writer.paragraph("Course code CS-304. Weightage in end-semester examination: 22 marks. Recommended revision time: 3 hours.")

    writer.heading("3.1 FUNCTIONAL DEPENDENCY - DEFINITION")
    writer.paragraph(
        "A functional dependency X -> Y holds on a relation R if and only if, for every pair of tuples "
        "t1 and t2 in R, whenever t1[X] = t2[X] it is also true that t1[Y] = t2[Y]. X is called the "
        "determinant and Y the dependent. A dependency is trivial when Y is a subset of X."
    )
    writer.paragraph(
        "Armstrong's axioms are sound and complete for deriving all functional dependencies implied by a "
        "given set F. The three primary rules are reflexivity, augmentation and transitivity; the three "
        "derived rules are union, decomposition and pseudo-transitivity."
    )
    writer.bullet("Reflexivity: if Y is a subset of X then X -> Y.")
    writer.bullet("Augmentation: if X -> Y then XZ -> YZ for any attribute set Z.")
    writer.bullet("Transitivity: if X -> Y and Y -> Z then X -> Z.")
    writer.bullet("Union: if X -> Y and X -> Z then X -> YZ.")
    writer.bullet("Decomposition: if X -> YZ then X -> Y and X -> Z.")
    writer.bullet("Pseudo-transitivity: if X -> Y and WY -> Z then WX -> Z.")

    writer.heading("3.2 ATTRIBUTE CLOSURE AND CANDIDATE KEYS")
    writer.paragraph(
        "The closure of an attribute set X, written X+, is the set of all attributes functionally "
        "determined by X under F. X is a superkey of R if and only if X+ equals the full set of "
        "attributes of R. X is a candidate key if it is a superkey and no proper subset of X is a "
        "superkey, that is, the key is irreducible."
    )
    writer.paragraph(
        "Worked example. Let R(A, B, C, D, E) with F = {A -> BC, CD -> E, B -> D, E -> A}. Then the "
        "closure of A is A+ = {A, B, C, D, E}, so A is a superkey and, being a single attribute, also a "
        "candidate key. Similarly E -> A gives E+ = {A, B, C, D, E}, so E is a candidate key. CD is also "
        "a candidate key because CD -> E and E -> A. The prime attributes are therefore A, B, C, D and E "
        "is prime as well; BC is not a key because BC+ = {B, C, D, E, A} - verify by derivation."
    )

    writer.heading("3.3 NORMAL FORMS - CONDITIONS AND COMPARISON")
    writer.row(["Normal form", "Condition to satisfy", "Removes"], [122.0, 210.0, 130.0], bold=True)
    writer.row(["1NF", "All attribute values are atomic", "Repeating groups"], [122.0, 210.0, 130.0])
    writer.row(["2NF", "1NF and no partial dependency", "Partial dependency"], [122.0, 210.0, 130.0])
    writer.row(["3NF", "2NF and no transitive dependency", "Transitive dependency"], [122.0, 210.0, 130.0])
    writer.row(["BCNF", "Every determinant is a superkey", "All key anomalies"], [122.0, 210.0, 130.0])
    writer.row(["4NF", "BCNF and no multivalued dependency", "MVD redundancy"], [122.0, 210.0, 130.0])
    writer.row(["5NF", "4NF and no join dependency", "Join redundancy"], [122.0, 210.0, 130.0])
    writer.spacer()
    writer.paragraph(
        "A relation is in second normal form if it is in first normal form and every non-prime attribute "
        "is fully functionally dependent on every candidate key. A relation is in third normal form if, "
        "for every non-trivial dependency X -> A, either X is a superkey or A is a prime attribute."
    )
    writer.paragraph(
        "A relation is in Boyce-Codd normal form if for every non-trivial functional dependency X -> Y "
        "the determinant X is a superkey of the relation. BCNF is strictly stronger than 3NF: every BCNF "
        "relation is in 3NF, but a 3NF relation with overlapping candidate keys need not be in BCNF."
    )
    writer.paragraph(
        "Important trade-off frequently asked in examinations: a decomposition into 3NF that is both "
        "lossless and dependency preserving always exists, whereas a decomposition into BCNF is always "
        "lossless but may fail to preserve all functional dependencies. The classic counter-example is "
        "R(Student, Subject, Teacher) with Student, Subject -> Teacher and Teacher -> Subject."
    )

    writer.heading("3.4 LOSSLESS JOIN AND DEPENDENCY PRESERVATION TESTS")
    writer.paragraph(
        "A binary decomposition of R into R1 and R2 is lossless if and only if the intersection of R1 and "
        "R2 is a superkey of R1 or a superkey of R2. Equivalently, either (R1 intersect R2) -> R1 or "
        "(R1 intersect R2) -> R2 must hold in F+."
    )
    writer.paragraph(
        "A decomposition preserves dependencies if the union of the projections of F onto each fragment "
        "is equivalent to F, that is, the closure of the union equals F+. For n fragments, use the "
        "chase algorithm or the tableau method to test the lossless join property."
    )

    writer.heading("3.5 TRANSACTIONS AND THE ACID PROPERTIES")
    writer.bullet("Atomicity: a transaction executes entirely or not at all; enforced by the recovery manager using the undo log.")
    writer.bullet("Consistency: a transaction takes the database from one valid state to another valid state, preserving all declared integrity constraints.")
    writer.bullet("Isolation: concurrent transactions must produce a result equivalent to some serial execution; enforced by the concurrency control manager.")
    writer.bullet("Durability: once a transaction commits, its effects survive any subsequent system failure; enforced by write-ahead logging and forced log flush at commit.")

    writer.heading("3.6 ISOLATION LEVELS AND CONCURRENCY ANOMALIES")
    writer.row(["Isolation level", "Dirty read", "Non-repeatable read", "Phantom read"], [150.0, 100.0, 130.0, 90.0], bold=True)
    writer.row(["Read Uncommitted", "Possible", "Possible", "Possible"], [150.0, 100.0, 130.0, 90.0])
    writer.row(["Read Committed", "Prevented", "Possible", "Possible"], [150.0, 100.0, 130.0, 90.0])
    writer.row(["Repeatable Read", "Prevented", "Prevented", "Possible"], [150.0, 100.0, 130.0, 90.0])
    writer.row(["Serializable", "Prevented", "Prevented", "Prevented"], [150.0, 100.0, 130.0, 90.0])
    writer.spacer()
    writer.paragraph(
        "A dirty read occurs when a transaction reads data written by an uncommitted transaction. A "
        "non-repeatable read occurs when a transaction reads the same row twice and obtains different "
        "values. A phantom read occurs when a transaction re-executes a range query and finds newly "
        "committed rows that satisfy the predicate."
    )

    writer.heading("3.7 LOCKING PROTOCOLS")
    writer.paragraph(
        "Two-phase locking (2PL) requires that all lock acquisitions precede all lock releases. The "
        "growing phase acquires locks and the shrinking phase releases them. Basic 2PL guarantees "
        "serializability but not freedom from cascading rollback."
    )
    writer.bullet("Strict 2PL: all exclusive locks are held until commit or abort; avoids cascading rollback.")
    writer.bullet("Rigorous 2PL: all locks, shared and exclusive, are held until commit; simplifies recovery.")
    writer.bullet("Conservative 2PL: all locks are acquired before the transaction begins; deadlock-free but rarely practical.")
    writer.paragraph(
        "Deadlock is detected by constructing a wait-for graph and testing for a cycle; it is prevented "
        "by the wait-die scheme, in which an older transaction waits and a younger one aborts, or by the "
        "wound-wait scheme, in which an older transaction pre-empts a younger one."
    )

    writer.heading("3.8 RECOVERY - WRITE AHEAD LOGGING")
    writer.paragraph(
        "The write-ahead logging rule states that a log record describing a change must reach stable "
        "storage before the corresponding data page is written to disk. Two consequences follow: undo "
        "information must be logged before an uncommitted update is flushed, and redo information must "
        "be logged before a commit is acknowledged."
    )
    writer.paragraph(
        "During restart recovery the ARIES algorithm performs three passes: an analysis pass that "
        "identifies the winners and losers from the most recent checkpoint, a redo pass that repeats "
        "history for all updates, and an undo pass that rolls back the losers in reverse order using "
        "compensation log records."
    )

    writer.heading("3.9 FORMULAS WORTH MEMORISING")
    writer.bullet("Maximum number of superkeys of a relation with n attributes and a single-attribute candidate key: 2^(n-1).")
    writer.bullet("Number of possible functional dependencies over n attributes: 2^n multiplied by 2^n, that is 4^n including trivial ones.")
    writer.bullet("Lossless binary decomposition test: (R1 intersect R2) must be a superkey of R1 or of R2.")
    writer.bullet("Cost of a block nested loop join: B(R) + B(R) multiplied by B(S) divided by (M - 2) blocks.")

    writer.heading("3.10 PREVIOUSLY ASKED EXAMINATION QUESTIONS")
    writer.bullet("Define BCNF and prove that every BCNF relation is in 3NF. (6 marks, asked in 2021 and 2023)")
    writer.bullet("Given R(A,B,C,D,E) and F, compute all candidate keys and normalise up to BCNF. (10 marks)")
    writer.bullet("Differentiate between conflict serializability and view serializability with an example. (6 marks)")
    writer.bullet("Explain why Repeatable Read still permits phantom reads and how Serializable prevents them. (5 marks)")
    writer.bullet("State the write-ahead logging rule and describe the three passes of ARIES recovery. (8 marks)")

    return writer.save(out_dir / "study_notes_dbms_unit3.pdf")


# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the VeriRAG sample PDF corpus.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = (build_case_law, build_lease_deed, build_study_notes)
    for builder in builders:
        path = builder(out_dir)
        with fitz.open(path) as doc:
            pages = doc.page_count
        size_kb = path.stat().st_size / 1024
        print(f"  created {path.name:<40} {pages} pages  {size_kb:6.1f} KB")

    print(f"\n{len(builders)} sample PDFs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
