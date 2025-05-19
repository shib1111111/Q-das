from jinja2 import Environment, FileSystemLoader
import pdfkit
from datetime import datetime
import pytz
import os

def render_report_to_pdf(metadata: dict, parameter_info: dict, output_pdf: str = 'temp_file/mca_cmm_report.pdf') -> str:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(current_dir, 'templates')
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template('empty_reporting.html')

        ist = pytz.timezone('Asia/Kolkata')
        current_date = datetime.now(ist).strftime('%d/%m/%Y')
        current_datetime = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')

        temp_html = 'temp_report.html'
        combined_html = ""

        total_pages = len(parameter_info)
        logo_path = os.path.abspath(os.path.join(current_dir, 'static', 'cdac_logo.png'))
        for index, (param_name, param) in enumerate(parameter_info.items(), 1):
            html_content = template.render(
                metadata=metadata,
                param=param,
                param_name=param_name,
                current_date=current_date,
                current_datetime=current_datetime,
                page_number=index,
                total_pages=total_pages,
                logo_path=f"file://{logo_path}"
            )
            combined_html += html_content
            # Add page break only if this is not the last page
            if index < total_pages:
                combined_html += '<div style="page-break-after: always;"></div>'

        try:
            with open(temp_html, 'w', encoding='utf-8') as f:
                f.write(combined_html)
        except IOError as e:
            raise Exception(f"Error writing temporary HTML file: {str(e)}")

        options = {
            'page-size': 'A4',
            'margin-top': '10mm',    # 1 cm
            'margin-bottom': '10mm', # 1 cm
            'margin-left': '10mm',   # 1 cm
            'margin-right': '10mm',  # 1 cm
            'encoding': 'UTF-8',
            'dpi': 300,
            'enable-local-file-access': ''
        }

        # Ensure output directory exists
        if os.path.dirname(output_pdf):
            os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

        # Use the confirmed wkhtmltopdf path
        pdfkit.from_file(temp_html, output_pdf, options=options)

        if os.path.exists(temp_html):
            os.remove(temp_html)

        return output_pdf

    except Exception as e:
        raise Exception(f"Error generating PDF report: {str(e)}")