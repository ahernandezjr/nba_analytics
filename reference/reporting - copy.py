from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from datetime import datetime
import os

# Assuming settings are similar to those in reporting.py
from .config import settings

DATA_DIR = settings.DATA_DIR
GRAPHS_DIR = settings.GRAPHS_DIR
REPORTS_DIR = settings.REPORTS_DIR

TITLE = "Weekly Analytics Report"
WIDTH = 210 * mm

def create_letterhead(c, width):
    c.drawImage(os.path.join(REPORTS_DIR, "letterhead.png"), 0, 0, width=width, preserveAspectRatio=True)


def create_title(c, title):
    c.setFont("Times-Bold", 20)
    c.drawString(10 * mm, 280 * mm, title)
    today = datetime.now().strftime("%d/%m/%Y")
    c.setFont("Times-Roman", 14)
    c.drawString(20 * mm, 275 * mm, f" - ({today})")

    # New line
    c.drawString(10 * mm, 270 * mm, "")


def create_section(c, title):
    c.setFont("Times-Bold", 16)
    c.drawString(20 * mm, 10 * mm, title)
    c.drawString(10 * mm, 270 * mm, "")


def write_to_pdf(c, text):
    c.drawString(10 * mm, 270 * mm, text)


def end_section(c):
    c.drawString(10 * mm, 270 * mm, "")


def add_image(c, image_path, width=170 * mm):
    c.drawImage(image_path, 10 * mm, - 60 * mm, width=width, preserveAspectRatio=True)
    c.showPage()
    c.drawString(10 * mm, 270 * mm, "")


def footer(canvas):
    canvas.saveState()
    canvas.setFont('Times-Italic', 8)
    canvas.drawString(A4[0] - 100, 10, f"Page {canvas.getPageNumber}")
    canvas.restoreState()


def create_report():
    c = canvas.Canvas(os.path.join(REPORTS_DIR, "EXAMPLE_report.pdf"), pagesize=A4)
    c.setAuthor("Your Name")
    
    # Introduction Page
    c.setFont("Times-Roman", 12)
    create_title(c, TITLE)
    create_section(c, "Table of Contents")
    # Add more sections as needed
    
    # Adding an image
    analytics_image_path = os.path.join(GRAPHS_DIR, "analytics.png")
    add_image(c, analytics_image_path)
    
    # Ensure to call showPage() to add a new page
    c.showPage()
    
    create_section("Table of Contents")
    write_to_pdf(c, "- [Goals](#goals)")
    # Hyperlink to the Key Features section


    write_to_pdf(c, "- [Key Features](#key-features)")
    write_to_pdf(c, "- [Analytics](#analytics)")
    write_to_pdf(c, "- [Data Specifications](#data-specifications)")
    write_to_pdf(c, "- [Insights and Analysis](#insights-and-analysis)")
    end_section(c)

    # Add Goals section
    create_section(c, "Goals")
    write_to_pdf(c, "- Primary goal: Determine which players on the Detroit Pistons are valuable after 1 season of play.")
    write_to_pdf(c, "- Secondary goals:")
    write_to_pdf(c, "  - Understand the data and perform analytics on its specifications and data insights.")
    write_to_pdf(c, "  - Perform basic statistical modeling on the data.")
    write_to_pdf(c, "  - Use neural nets like MLP, ARIMA, and LSTM to predict what to do with the players.")
    end_section(c)

    # Add Key Features section
    create_section(c, "Key Features")
    write_to_pdf(c, "- Collect and analyze data on Detroit Pistons players.")
    write_to_pdf(c, "- Perform statistical modeling and analysis on the data.")
    write_to_pdf(c, "- Use neural networks like MLP, ARIMA, and LSTM for player predictions.")
    write_to_pdf(c, "- Generate insights and analysis based on the data.")
    end_section(c)

    '''
    First Page of PDF
    '''
    # Add Page
    c.showPage()
    create_letterhead(c, WIDTH)

    # Add lettterhead and title
    # create_letterhead(pdf, WIDTH)
    
    # Add Analytics section
    create_section(c, "Analytics")
    write_to_pdf(c, "![Graph](data/graphs/analytics.png)")
    write_to_pdf(c, "Data Specifications:")
    write_to_pdf(c, f"- Original DataFrame: Entries={13210}, Unique Players={2377}")
    write_to_pdf(c, f"- Filtered DataFrame: Entries={950}, Unique Players={190}")
    write_to_pdf(c, "Insights and Analysis:")
    write_to_pdf(c, "- [Insight 1]")
    write_to_pdf(c, "- [Insight 2]")
    write_to_pdf(c, "- [Insight 3]")
    end_section(c)


    # Add some words to PDF
    write_to_pdf(c, "1. The graph below demonstates the basic analytics of the NBA dataset:")

    # Add table
    c.image(os.path.join(GRAPHS_DIR, "analytics.png"), w=170)
    c.ln(10)

    # First page content text
    write_to_pdf(c, "2. The visualisations below show model prediction comparisons:")

    # Add the generated visualisations to the PDF
    c.image(os.path.join(GRAPHS_DIR, "model_predictions.png"), 5, 200, WIDTH/2-10)
    # pdf.image(os.path.join(GRAPHS_DIR, "pca.png"), WIDTH/2, 200, WIDTH/2-10)
    c.ln(10)

    # Footer
    footer(c)
    
    c.save()

if __name__ == "__main__":
    create_report()