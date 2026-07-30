from pathlib import Path

from faker import Faker
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from decimal import Decimal, ROUND_HALF_UP

from .common import render_pdf_to_png


MONEY_STEP = Decimal("0.01")

PRODUCT_ADJECTIVES = [
    "Draadloze",
    "Ergonomische",
    "Compacte",
    "Professionele",
    "Verstelbare",
    "Premium",
]

PRODUCT_NOUNS = [
    "muis",
    "toetsenbord",
    "monitorstandaard",
    "USB-C-kabel",
    "webcam",
    "laptophoes",
    "bureauhouder",
    "adapter",
]


def as_money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_STEP, rounding=ROUND_HALF_UP)


def format_euro(value: Decimal) -> str:
    return f"€ {value:.2f}".replace(".", ",")


def generate_items(fake: Faker) -> list[dict]:
    items = []

    item_count = fake.random_int(min=3, max=7)

    for _ in range(item_count):
        product = (
            f"{fake.random_element(PRODUCT_ADJECTIVES)} "
            f"{fake.random_element(PRODUCT_NOUNS)}"
        )

        quantity = fake.random_int(min=1, max=8)

        unit_price_excl = as_money(
            Decimal(fake.random_int(min=500, max=25000)) / Decimal("100")
        )

        # Demo-btw-percentages voor gevarieerde testdata.
        vat_rate = Decimal(
            str(fake.random_element(elements=(6, 12, 21)))
        )

        subtotal_excl = as_money(
            unit_price_excl * Decimal(quantity)
        )

        vat_amount = as_money(
            subtotal_excl * vat_rate / Decimal("100")
        )

        total_incl = as_money(
            subtotal_excl + vat_amount
        )

        items.append(
            {
                "product": product,
                "quantity": quantity,
                "unit_price_excl": unit_price_excl,
                "vat_rate": vat_rate,
                "subtotal_excl": subtotal_excl,
                "vat_amount": vat_amount,
                "total_incl": total_incl,
            }
        )

    return items

def generate_receipt(
    output_dir: Path,
    index: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    fake = Faker("nl_BE")
    fake.seed_instance(42 + index)

    document_id = f"receipt_{index:03d}"

    pdf_path = output_dir / f"{document_id}.pdf"
    png_path = output_dir / f"{document_id}.png"

    items = generate_items(fake)

    subtotal_excl = as_money(
        sum(
            (item["subtotal_excl"] for item in items),
            start=Decimal("0.00"),
        )
    )

    total_vat = as_money(
        sum(
            (item["vat_amount"] for item in items),
            start=Decimal("0.00"),
        )
    )

    grand_total = as_money(subtotal_excl + total_vat)

    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4

    # Titel
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(40, height - 50, "KASSABON")

    # Leverancier
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(40, height - 85, "Leverancier")

    pdf.setFont("Helvetica", 9)
    pdf.drawString(40, height - 102, fake.company())
    pdf.drawString(
        40,
        height - 117,
        fake.address().replace("\n", ", "),
    )

    # Kassaboninformatie
    receipt_number = fake.random_number(digits=6)
    receipt_date = fake.date_this_year()

    pdf.setFont("Helvetica-Bold", 10)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(350, height - 85, "Kassabongegevens")

    pdf.setFont("Helvetica", 9)
    pdf.drawString(
        350,
        height - 102,
        f"Kassabonnummer: {receipt_number}",
    )
    pdf.drawString(
        350,
        height - 117,
        f"Datum: {receipt_date}",
    )

    # Tabelinstellingen
    table_top = height - 180
    row_height = 24

    x_product = 40
    x_quantity = 280
    x_unit_price = 330
    x_subtotal = 395
    x_vat_rate = 450
    x_vat_amount = 500
    x_total = 565

    # Tabelheader
    pdf.setFillGray(0.90)
    pdf.rect(
        35,
        table_top - 5,
        width - 60,
        row_height,
        fill=1,
        stroke=0,
    )
    pdf.setFillGray(0)

    pdf.setFont("Helvetica-Bold", 7.5)
    pdf.drawString(x_product, table_top + 4, "Product")
    pdf.drawRightString(x_quantity, table_top + 4, "Aantal")
    pdf.drawRightString(x_unit_price, table_top + 4, "Prijs excl.")
    pdf.drawRightString(x_subtotal, table_top + 4, "Subtotaal")
    pdf.drawRightString(x_vat_rate, table_top + 4, "Btw")
    pdf.drawRightString(x_vat_amount, table_top + 4, "Btw-bedrag")
    pdf.drawRightString(x_total, table_top + 4, "Totaal incl.")

    # Factuurlijnen
    current_y = table_top - row_height

    pdf.setFont("Helvetica", 7.5)

    for item in items:
        pdf.setStrokeGray(0.80)
        pdf.line(
            35,
            current_y - 5,
            width - 25,
            current_y - 5,
        )

        pdf.setFillGray(0)

        pdf.drawString(
            x_product,
            current_y + 4,
            item["product"][:34],
        )
        pdf.drawRightString(
            x_quantity,
            current_y + 4,
            str(item["quantity"]),
        )
        pdf.drawRightString(
            x_unit_price,
            current_y + 4,
            format_euro(item["unit_price_excl"]),
        )
        pdf.drawRightString(
            x_subtotal,
            current_y + 4,
            format_euro(item["subtotal_excl"]),
        )
        pdf.drawRightString(
            x_vat_rate,
            current_y + 4,
            f"{item['vat_rate']:.0f}%",
        )
        pdf.drawRightString(
            x_vat_amount,
            current_y + 4,
            format_euro(item["vat_amount"]),
        )
        pdf.drawRightString(
            x_total,
            current_y + 4,
            format_euro(item["total_incl"]),
        )

        current_y -= row_height

    # Totalenblok
    summary_y = current_y - 25
    label_x = 410
    value_x = 565

    pdf.setFont("Helvetica", 9)
    pdf.drawRightString(
        label_x,
        summary_y,
        "Subtotaal excl. btw:",
    )
    pdf.drawRightString(
        value_x,
        summary_y,
        format_euro(subtotal_excl),
    )

    pdf.drawRightString(
        label_x,
        summary_y - 18,
        "Totaal btw:",
    )
    pdf.drawRightString(
        value_x,
        summary_y - 18,
        format_euro(total_vat),
    )

    pdf.setStrokeGray(0.3)
    pdf.line(
        label_x - 20,
        summary_y - 28,
        value_x,
        summary_y - 28,
    )

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawRightString(
        label_x,
        summary_y - 45,
        "Te betalen:",
    )
    pdf.drawRightString(
        value_x,
        summary_y - 45,
        format_euro(grand_total),
    )

    # Footer
    pdf.setFont("Helvetica-Oblique", 7)
    pdf.setFillGray(0.35)
    pdf.drawString(
        40,
        35,
        "Synthetisch document voor test- en demonstratiedoeleinden.",
    )

    pdf.save()

    # PDF naar PNG renderen
    render_pdf_to_png(
        pdf_path=pdf_path,
        png_path=png_path,
        scale=2.0,
    )

    return {
        "id": document_id,
        "path": png_path.as_posix(),
        "source": "synthetic",
        "document_type": "receipt",
        "language": "nl",
        "contains_table": True,
        "contains_signature": False,
        "contains_logo": False,
        "product_row_count": len(items),
        "subtotal_excl": str(subtotal_excl),
        "total_vat": str(total_vat),
        "grand_total": str(grand_total),
        "safe_for_public_repo": True,
        "seed": 42 + index,
    }