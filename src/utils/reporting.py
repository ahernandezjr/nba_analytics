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

# Global Variables
TITLE = "Weekly Analytics Report"
WIDTH = 210
HEIGHT = 297

MARGIN_SIZE = 10
TITLE_SIZE = 24
SECTION_SIZE = 16
TEXT_SIZE = 12


# Change from fpdf to reportlab
class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')


def create_letterhead(pdf, WIDTH):
    pdf.image(os.path.join(REPORTS_DIR, "extra_files", "letterhead.png"), 0, 0, WIDTH)
    pdf.ln(25)


def end_section(pdf):
    pdf.ln(10)


def write_to_pdf(pdf, words, link=None):
    # Set text colour, font size, and font type
    pdf.set_font('Arial', '', TEXT_SIZE)
    pdf.set_left_margin(MARGIN_SIZE + 10)
    pdf.ln(7)
    
    pdf.write(5, words, link)


def create_image(pdf, image_path):
    extra_margin = 10
    pdf.set_left_margin(MARGIN_SIZE)
    pdf.ln(5)
    pdf.image(image_path, x=MARGIN_SIZE + extra_margin, w=WIDTH - (MARGIN_SIZE*2 + extra_margin*2))
    pdf.ln(5)
    

def generate_commits_png():
    # Get the git commits
    repo = git.Repo(search_parent_directories=True)
    commits = list(repo.iter_commits(max_count=7))

    # Create a DataFrame of the commits
    commit_data = pd.DataFrame([(commit.hexsha, commit.message, commit.author.name, commit.authored_datetime) for commit in commits], columns=["Hash", "Message", "Author", "Date"])
    commit_data = commit_data.drop(columns=["Hash", "Author"])
    # change type of Date column to datetime
    commit_data['Date'] = pd.to_datetime(commit_data['Date'], utc=True).dt.date

    # dt.date to pd.Timestamp
    commit_data['Date'] = pd.to_datetime(commit_data['Date'])

    # Keep all rows with the date in the last 7 days
    commit_data = commit_data[commit_data['Date'] >= pd.Timestamp.today() - pd.DateOffset(days=7)]
   
    # To do: show full commit message

    # If the extra_files directory does not exist, create it
    if not os.path.exists(os.path.join(REPORTS_DIR, "extra_files")):
        os.makedirs(os.path.join(REPORTS_DIR, "extra_files"))
    
    dfi.export(commit_data, os.path.join(REPORTS_DIR, "extra_files", "commits.png"))


def create_cover(pdf, title, date, text):
    # Add main title, centered
    pdf.set_font('Times', 'b', TITLE_SIZE)
    pdf.ln(55)

    pdf.cell(w=0, h=10, txt=title, border=0, ln=1, align='C')

    pdf.set_font('Times', 'i', SECTION_SIZE)
    pdf.cell(w=0, h=10, txt=date, border=0, ln=1, align='C')
    pdf.ln(10)

    # Add text
    pdf.set_font('Times', '', SECTION_SIZE)
    pdf.set_left_margin(MARGIN_SIZE + 15)
    pdf.set_right_margin(MARGIN_SIZE + 15)
    pdf.multi_cell(w=0, h=10, txt=text, border=0, align='C')
    pdf.ln(10)


def create_section(pdf, title):
    pdf.set_font('Times', 'b', SECTION_SIZE)
    pdf.set_text_color(r=0,g=0,b=0)
    pdf.set_left_margin(MARGIN_SIZE)
    pdf.ln(10)

    pdf.write(5, title)


def create_intro_pages(pdf):
    '''
    Cover Page of PDF
    '''
    pdf.add_page()
    create_letterhead(pdf, WIDTH)

    # Get date
    today = time.strftime("%d/%m/%Y")
    cover_desc = "This is a project for organizing data collection, data processing, and machine learning tasks related to NBA player statistics, specifically to determine valuable players among the DETROIT PISTONS."

    # Add Table of Contents
    create_cover(pdf, TITLE, today, cover_desc)

    '''
    Introduction Page of PDF
    '''
    pdf.add_page()
    create_letterhead(pdf, WIDTH)

    # Hyperlinks to sections
    to_goals = pdf.add_link()
    to_key_features = pdf.add_link()
    to_analytics = pdf.add_link()
    to_data_specs = pdf.add_link()
    to_insights_and_analysis = pdf.add_link()
    pdf.set_link(to_goals, page=2)
    pdf.set_link(to_key_features, page=2)
    pdf.set_link(to_analytics, page=3)
    pdf.set_link(to_data_specs, page=4)
    pdf.set_link(to_insights_and_analysis, page=5)

    # Add Table of Contents
    create_section(pdf, "Table of Contents")
    write_to_pdf(pdf, "- Goals", to_goals)
    write_to_pdf(pdf, "- Key Features", to_key_features)
    write_to_pdf(pdf, "- Analytics", to_analytics)
    write_to_pdf(pdf, "- Data Specifications", to_data_specs)
    write_to_pdf(pdf, "- Insights and Analysis", to_insights_and_analysis)
    end_section(pdf)

    # Add Goals section
    create_section(pdf, "Goals")
    write_to_pdf(pdf, "- Primary goal: Determine which players on the Detroit Pistons are valuable after 1 season of play.")
    write_to_pdf(pdf, "- Secondary goals:")
    write_to_pdf(pdf, "  - Understand the data and perform analytics on its specifications and data insights.")
    write_to_pdf(pdf, "  - Perform basic statistical modeling on the data.")
    write_to_pdf(pdf, "  - Use neural nets like MLP, ARIMA, and LSTM to predict what to do with the players.")
    end_section(pdf)

    # Add Key Features section
    create_section(pdf, "Key Features")
    write_to_pdf(pdf, "- Collect and analyze data on Detroit Pistons players.")
    write_to_pdf(pdf, "- Perform statistical modeling and analysis on the data.")
    write_to_pdf(pdf, "- Use neural networks like MLP, ARIMA, and LSTM for player predictions.")
    write_to_pdf(pdf, "- Generate insights and analysis based on the data.")
    end_section(pdf)


def create_report():
    # Create PDF
    pdf = PDF() # A4 (210 by 297 mm)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(MARGIN_SIZE)
    pdf.set_right_margin(MARGIN_SIZE)

    # Create intro pages
    create_intro_pages(pdf)
    
    '''
    First Page of PDF
    '''
    # Add Page
    pdf.add_page()
    create_letterhead(pdf, WIDTH)

    # Add Analytics section
    create_section(pdf, "Analytics")
    write_to_pdf(pdf, "Data Specifications:")
    write_to_pdf(pdf, f"- Original DataFrame: Entries={13210}, Unique Players={2377}")
    write_to_pdf(pdf, f"- Filtered DataFrame: Entries={950}, Unique Players={190}")
    write_to_pdf(pdf, "Insights and Analysis:")
    write_to_pdf(pdf, "- [Insight 1]")
    write_to_pdf(pdf, "- [Insight 2]")
    write_to_pdf(pdf, "- [Insight 3]")
    end_section(pdf)


    # First page content text
    write_to_pdf(pdf, "1. The graph below demonstates the basic analytics of the NBA dataset:")

    # Add the generated visualisations to the PDF
    create_image(pdf, os.path.join(GRAPHS_DIR, "analytics.png"))
    pdf.ln(10)

    # First page content text
    write_to_pdf(pdf, "2. The visualisations below show model prediction comparisons:")

    # Add the generated visualisations to the PDF
    create_image(pdf, os.path.join(GRAPHS_DIR, "model_predictions.png"))
    # pdf.image(os.path.join(GRAPHS_DIR, "pca.png"), WIDTH/2, 200, WIDTH/2-10)
    pdf.ln(10)


    '''
    Second Page of PDF
    '''

    # Add Page
    pdf.add_page()
    create_letterhead(pdf, WIDTH)

    # Second page content text
    write_to_pdf(pdf, "3. The graphs below show further analysis via PCA, showing dataset sample relationships.")

    # Add the generated visualisations to the PDF
    create_image(pdf, os.path.join(GRAPHS_DIR, "pca.png"))

    '''
    Third Page of PDF
    '''

    # Add Page
    pdf.add_page()
    create_letterhead(pdf, WIDTH)

    create_section(pdf, "Productivity")

    # Add some words to PDF
    # Get all git commits from the last 7 days
    write_to_pdf(pdf, "The table below shows the last 7 days of git commits:")

    # Generate the commits table
    generate_commits_png()

    # Add the table to the PDF
    create_image(pdf, os.path.join(REPORTS_DIR, "extra_files", "commits.png"))
    pdf.ln(10)


    # Get the current date
    today = time.strftime("%Y-%m-%d")

    # If report directory does not exist, create it
    if not os.path.exists(os.path.join(DATA_DIR, "reports")):
        os.makedirs(os.path.join(DATA_DIR, "reports"))
    # Generate the PDF
    # pdf.output(os.path.join(REPORTS_DIR, f"EXAMPLE_{today}_report.pdf"), 'F')
    pdf.output(os.path.join(REPORTS_DIR, f"{today}_report.pdf"), 'F')


if __name__ == "__main__":
    create_report()

