import fpdf
from fpdf import FPDF
import time
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi

import git

from .config import settings
from .logger import get_logger
import os


# Create logger
logger = get_logger(__name__)


# Set configs from settings
DATA_DIR = settings.DATA_DIR
GRAPHS_DIR = settings.GRAPHS_DIR
REPORTS_DIR = settings.REPORTS_DIR



# generate_matplotlib_stackbars(df, 'resources/heicoders_annual_sales.png')
# generate_matplotlib_piechart(df, 'resources/heicoders_2016_sales_breakdown.png')


def create_letterhead(pdf, WIDTH):
    pdf.image(DATA_DIR + "/example_letterhead.png", 0, 0, WIDTH)

def create_title(title, pdf):
    # Add main title
    pdf.set_font('Helvetica', 'b', 20)  
    pdf.ln(40)
    pdf.write(5, title)
    pdf.ln(10)
    
    # Add date of report
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(r=128,g=128,b=128)
    today = time.strftime("%d/%m/%Y")
    pdf.write(4, f'{today}')
    
    # Add line break
    pdf.ln(10)

def write_to_pdf(pdf, words):
    
    # Set text colour, font size, and font type
    pdf.set_text_color(r=0,g=0,b=0)
    pdf.set_font('Helvetica', '', 12)
    
    pdf.write(5, words)


def generate_commits_png():
    # Get the git commits
    repo = git.Repo(search_parent_directories=True)
    commits = list(repo.iter_commits(max_count=7))

    # Create a DataFrame of the commits
    commit_data = pd.DataFrame([(commit.hexsha, commit.message, commit.author.name, commit.authored_datetime) for commit in commits], columns=["Hash", "Message", "Author", "Date"])
    commit_data = commit_data.drop(columns=["Hash", "Author"])

    # If the extra_files directory does not exist, create it
    if not os.path.exists(os.path.join(REPORTS_DIR, "extra_files")):
        os.makedirs(os.path.join(REPORTS_DIR, "extra_files"))
    
    dfi.export(commit_data, os.path.join(REPORTS_DIR, "extra_files", "commits.png"))


class PDF(FPDF):

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')


# Global Variables
TITLE = "Weekly Business Report"
WIDTH = 210
HEIGHT = 297

def create_report():
    # Create PDF
    pdf = PDF() # A4 (210 by 297 mm)


    '''
    Introduction Page of PDF
    '''
    pdf.add_page()

    # Add Table of Contents
    create_title("Table of Contents", pdf)
    pdf.ln(10)
    write_to_pdf(pdf, "- [Goals](#goals)")
    pdf.ln(5)
    write_to_pdf(pdf, "- [Key Features](#key-features)")
    pdf.ln(5)
    write_to_pdf(pdf, "- [Analytics](#analytics)")
    pdf.ln(5)
    write_to_pdf(pdf, "- [Data Specifications](#data-specifications)")
    pdf.ln(5)
    write_to_pdf(pdf, "- [Insights and Analysis](#insights-and-analysis)")
    pdf.ln(10)

    # Add Goals section
    create_title("Goals", pdf)
    pdf.ln(10)
    write_to_pdf(pdf, "- Primary goal: Determine which players on the Detroit Pistons are valuable after 1 season of play.")
    pdf.ln(5)
    write_to_pdf(pdf, "- Secondary goals:")
    pdf.ln(5)
    write_to_pdf(pdf, "  - Understand the data and perform analytics on its specifications and data insights.")
    pdf.ln(5)
    write_to_pdf(pdf, "  - Perform basic statistical modeling on the data.")
    pdf.ln(5)
    write_to_pdf(pdf, "  - Use neural nets like MLP, ARIMA, and LSTM to predict what to do with the players.")
    pdf.ln(10)

    # Add Key Features section
    create_title("Key Features", pdf)
    pdf.ln(10)
    write_to_pdf(pdf, "- Collect and analyze data on Detroit Pistons players.")
    pdf.ln(5)
    write_to_pdf(pdf, "- Perform statistical modeling and analysis on the data.")
    pdf.ln(5)
    write_to_pdf(pdf, "- Use neural networks like MLP, ARIMA, and LSTM for player predictions.")
    pdf.ln(5)
    write_to_pdf(pdf, "- Generate insights and analysis based on the data.")
    pdf.ln(10)

    '''
    First Page of PDF
    '''
    # Add Page
    pdf.add_page()

    # Add lettterhead and title
    # create_letterhead(pdf, WIDTH)
    
    # Add Analytics section
    create_title("Analytics", pdf)
    pdf.ln(10)
    write_to_pdf(pdf, "![Graph](data/graphs/analytics.png)")
    pdf.ln(10)
    write_to_pdf(pdf, "Data Specifications:")
    pdf.ln(5)
    write_to_pdf(pdf, f"- Original DataFrame: Entries={13210}, Unique Players={2377}")
    pdf.ln(5)
    write_to_pdf(pdf, f"- Filtered DataFrame: Entries={950}, Unique Players={190}")
    pdf.ln(10)
    write_to_pdf(pdf, "Insights and Analysis:")
    pdf.ln(5)
    write_to_pdf(pdf, "- [Insight 1]")
    pdf.ln(5)
    write_to_pdf(pdf, "- [Insight 2]")
    pdf.ln(5)
    write_to_pdf(pdf, "- [Insight 3]")
    pdf.ln(10)


    # Add some words to PDF
    write_to_pdf(pdf, "1. The graph below demonstates the basic analytics of the NBA dataset:")
    pdf.ln(15)

    # Add table
    pdf.image(os.path.join(GRAPHS_DIR, "analytics.png"), w=170)
    pdf.ln(10)

    # First page content text
    write_to_pdf(pdf, "2. The visualisations below show model prediction comparisons:")

    # Add the generated visualisations to the PDF
    pdf.image(os.path.join(GRAPHS_DIR, "model_predictions.png"), 5, 200, WIDTH/2-10)
    # pdf.image(os.path.join(GRAPHS_DIR, "pca.png"), WIDTH/2, 200, WIDTH/2-10)
    pdf.ln(10)


    '''
    Second Page of PDF
    '''

    # Add Page
    pdf.add_page()

    # Add lettterhead
    # create_letterhead(pdf, WIDTH)

    # Second page content text
    pdf.ln(40)
    write_to_pdf(pdf, "3. The graphs below show further analysis via PCA, showing dataset sample relationships.")

    # Add the generated visualisations to the PDF
    pdf.image(os.path.join(GRAPHS_DIR, "pca.png"), WIDTH/2, 200, WIDTH/2-10)
    pdf.ln(15)


    '''
    Third Page of PDF
    '''

    # Add Page
    pdf.add_page()

    # Add lettterhead
    # create_letterhead(pdf, WIDTH)

    # Add some words to PDF
    # Get all git commits from the last 7 days
    write_to_pdf(pdf, "4. The table below shows the last 7 days of git commits:")
    pdf.ln(15)

    # Generate the commits table
    generate_commits_png()

    # Add the table to the PDF
    pdf.image(os.path.join(REPORTS_DIR, "extra_files", "commits.png"), w=170)
    pdf.ln(10)


    # Get the current date
    today = time.strftime("%Y-%m-%d")

    # If report directory does not exist, create it
    if not os.path.exists(os.path.join(DATA_DIR, "reports")):
        os.makedirs(os.path.join(DATA_DIR, "reports"))
    # Generate the PDF
    # pdf.output(os.path.join(REPORTS_DIR, f"EXAMPLE_{today}_report.pdf"), 'F')
    pdf.output(os.path.join(REPORTS_DIR, f"EXAMPLE_report.pdf"), 'F')


if __name__ == "__main__":
    create_report()

