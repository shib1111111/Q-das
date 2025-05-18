from jinja2 import Environment, FileSystemLoader
import pdfkit
from datetime import datetime
import pytz
import os
path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

def render_report_to_pdf(metadata: dict, parameter_info: dict, output_pdf: str = 'temp_file/mca_cmm_report.pdf') -> str:
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # print(f"Current directory: {current_dir}")
        template_dir = os.path.join(current_dir,'templates')
        env = Environment(loader=FileSystemLoader(template_dir))        
        template = env.get_template('empty_reporting.html')

        ist = pytz.timezone('Asia/Kolkata')
        current_date = datetime.now(ist).strftime('%d/%m/%Y')
        current_datetime = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')

        temp_html = 'temp_report.html'
        combined_html = ""

        for index, (param_name, param) in enumerate(parameter_info.items(), 1):
            html_content = template.render(
                metadata=metadata,
                param=param,
                param_name=param_name,
                current_date=current_date,
                current_datetime=current_datetime,
                page_number=index,
                total_pages=len(parameter_info)
            )
            combined_html += html_content + '<div style="page-break-after: always;"></div>'

        with open(temp_html, 'w', encoding='utf-8') as f:
            f.write(combined_html)

        options = {
            'page-size': 'A4',
            'margin-top': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '20mm',
            'margin-right': '20mm',
            'encoding': 'UTF-8',
            'enable-local-file-access': ''
        }

        # ✅ Ensure output directory exists
        if os.path.dirname(output_pdf):
            os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

        pdfkit.from_file(temp_html, output_pdf, options=options,configuration=config)

        if os.path.exists(temp_html):
            os.remove(temp_html)

        return output_pdf

    except Exception as e:
        raise Exception(f"Error generating PDF report: {str(e)}")
