import io
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

class Exporter:
    @staticmethod
    def generate_txt(doc_data: dict) -> str:
        """Generates plain text export containing document header, Modi text, and Devanagari translation."""
        lines = [
            f"============================================================",
            f"MODI LIPI OCR & DIGITIZATION REPORT",
            f"Document ID: {doc_data.get('document_id')}",
            f"Filename: {doc_data.get('filename')}",
            f"============================================================\n",
            f"--- RECONSTRUCTED MODI TEXT ---",
            doc_data.get("modi_text", ""),
            f"\n--- DIGITIZED DEVANAGARI TEXT ---",
            doc_data.get("devanagari_text", ""),
            f"\n============================================================",
            f"STATISTICS:",
            f"Lines: {doc_data.get('statistics', {}).get('lines', 0)}",
            f"Words: {doc_data.get('statistics', {}).get('words', 0)}",
            f"Characters: {doc_data.get('statistics', {}).get('characters', 0)}",
            f"Average Confidence: {doc_data.get('statistics', {}).get('average_confidence', 0.0) * 100:.2f}%",
            f"Processing Time: {doc_data.get('statistics', {}).get('processing_time_ms', 0.0):.2f} ms",
            f"============================================================"
        ]
        return "\n".join(lines)

    @staticmethod
    def generate_json(doc_data: dict) -> str:
        """Generates formatted JSON export string."""
        return json.dumps(doc_data, indent=2, ensure_ascii=False)

    @staticmethod
    def generate_pdf(doc_data: dict) -> bytes:
        """Generates PDF file bytes using ReportLab."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#1E293B'),
            spaceAfter=12
        )
        section_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor('#2563EB'),
            spaceBefore=10,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            'BodyText',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#334155')
        )

        elements = []
        
        # Header Title
        elements.append(Paragraph("<b>Modi Lipi OCR & Digitization Report</b>", title_style))
        elements.append(Paragraph(f"<b>Document:</b> {doc_data.get('filename')} | <b>ID:</b> {doc_data.get('document_id')}", body_style))
        elements.append(Spacer(1, 10))

        # Statistics Table
        stats = doc_data.get("statistics", {})
        table_data = [
            ["Metric", "Value"],
            ["Total Lines", str(stats.get("lines", 0))],
            ["Total Words", str(stats.get("words", 0))],
            ["Total Characters", str(stats.get("characters", 0))],
            ["Average Confidence", f"{stats.get('average_confidence', 0.0) * 100:.2f}%"],
            ["Processing Time", f"{stats.get('processing_time_ms', 0.0):.2f} ms"],
            ["CNN Model Status", str(stats.get("model_status", "Active"))]
        ]
        
        t = Table(table_data, colWidths=[200, 250])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#F1F5F9')),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.HexColor('#0F172A')),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 15))

        # Modi Text Section
        elements.append(Paragraph("<b>Reconstructed Modi Lipi Text</b>", section_style))
        modi_text = doc_data.get("modi_text", "").replace("\n", "<br/>") or "<i>No Modi text detected</i>"
        elements.append(Paragraph(modi_text, body_style))
        elements.append(Spacer(1, 15))

        # Devanagari Digitized Text Section
        elements.append(Paragraph("<b>Digitized Devanagari Output</b>", section_style))
        dev_text = doc_data.get("devanagari_text", "").replace("\n", "<br/>") or "<i>No Devanagari output generated</i>"
        elements.append(Paragraph(dev_text, body_style))

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
