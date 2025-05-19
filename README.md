# Q-das: MCA/CMM Report Generator

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-darkblue)
![NumPy](https://img.shields.io/badge/NumPy-Array%20Computing-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-ScikitLearn-orange)
![wkhtmltopdf](https://img.shields.io/badge/wkhtmltopdf-0.12.6-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Q-das is a web-based application designed to generate MCA/CMM reports from Excel (.xlsx) input files. It processes inspection data, performs statistical analysis, and produces PDF reports with detailed parameter information and scatter plots, each parameter on a separate page.

## Features
- **Excel File Processing**: Upload Excel (.xlsx) files to extract metadata and parameter information.
- **Statistical Analysis**: Calculates process capability indices (Cp, Cpk), identifies best-fit distributions, and computes metrics like mean, median, standard deviation, and percentiles.
- **PDF Report Generation**: Creates multi-page PDF reports with parameter details, statistical metrics, scatter plots, and a logo.
- **Web Interface**: Simple HTML-based frontend for file uploads and report downloads.
- **Secure API**: FastAPI-based backend served over HTTPS with CORS support and SSL certificates.
- **Error Handling**: Robust handling for invalid inputs, file processing errors, and server issues.

## Project Structure
```
Q-das/
├── setup_env.sh
├── client/
│   └── index.html
├── server/
│   ├── api.py
│   ├── app.py
│   ├── requirements.txt
│   ├── app_utils/
│   │   ├── pdf_render.py
│   │   ├── utils.py
│   │   ├── static/
│   │   │   └── cdac_logo.png
│   │   ├── templates/
│   │   │   └── empty_reporting.html
│   ├── ca_certificates/
│   │   ├── cert.pem
│   │   ├── key.pem
│   ├── temp_file/
```

- **client/**: Contains the frontend HTML file (`index.html`).
- **server/**: Contains the FastAPI backend and utilities.
  - **api.py**: Defines API endpoints for metadata extraction and PDF generation.
  - **app.py**: Main FastAPI application with CORS middleware.
  - **app_utils/**: Utility scripts for data analysis and PDF rendering.
    - **utils.py**: Handles statistical analysis, scatter plot generation, and metadata parsing.
    - **pdf_render.py**: Renders HTML templates to PDF using Jinja2 and pdfkit.
    - **static/**: Stores static assets like the C-DAC logo.
    - **templates/**: Contains Jinja2 templates for PDF rendering.
  - **ca_certificates/**: SSL certificate and key for HTTPS.
  - **temp_file/**: Directory for temporary files and generated PDFs.
- **setup_env.sh**: Script to set up the virtual environment and install dependencies.
- **requirements.txt**: Lists Python dependencies for the backend.

## Prerequisites
- **Python 3.8+**: Required for the backend.
- **wkhtmltopdf**: Required for PDF generation (used by pdfkit).
- **Virtual Environment**: Recommended for dependency management.
- **Web Browser**: For accessing the web interface.
- **SSL Certificates**: Provided in `server/ca_certificates/` for HTTPS.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shib1111111/Q-das
   cd Q-das
   ```

2. **Set Up Virtual Environment**:
   - **On Linux/Mac**:
     Create a virtual environment and install dependencies:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     pip install -r server/requirements.txt
     ```
     Alternatively, use the provided setup script:
     ```bash
     chmod +x setup_env.sh
     ./setup_env.sh
     source venv/bin/activate
     ```
   - **On Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     pip install -r server/requirements.txt
     ```

3. **Install wkhtmltopdf**:
   - **Linux**:
     ```bash
     sudo apt-get install wkhtmltopdf
     ```
   - **Mac**:
     ```bash
     brew install wkhtmltopdf
     ```
   - **Windows**: Download and install from [wkhtmltopdf.org](https://wkhtmltopdf.org/downloads.html).

4. **Verify SSL Certificates**:
   Ensure `cert.pem` and `key.pem` are present in `server/ca_certificates/`. Replace with your own certificates if necessary.

## Running the Application

1. **Start the Backend Server**:
   From the `server/` directory, run:
   ```bash
   uvicorn app:app --host <ip> --port <port> --ssl-certfile ./ca_certificates/cert.pem --ssl-keyfile ./ca_certificates/key.pem
   ```
   Replace `<ip>` with the desired IP address (e.g., `0.0.0.0` for all interfaces) and `<port>` with the desired port (e.g., `8080`).
   The API will be available at `https://<ip>:<port>`.

   For running on localhost:
   ```bash
   uvicorn app:app --host localhost --port 8080 --ssl-certfile ./ca_certificates/cert.pem --ssl-keyfile ./ca_certificates/key.pem
   ```

2. **Access the Web Interface**:
   - Copy `client/index.html` to a web server directory (e.g., Nginx or Apache) or open it directly in a browser.
   - Alternatively, serve `index.html` using a simple HTTP server:
     ```bash
     cd client
     python -m http.server <port>
     ```
     Replace `<port>` with the desired port (e.g., `8000`).
     Access the interface at `http://<ip>:<port>` (e.g., `http://localhost:8000`).

3. **Generate a Report**:
   - Open the web interface in a browser.
   - Select an Excel (.xlsx) file (e.g., `DI REPORT OF DISC SEPERATOR.xlsx`).
   - Click "Upload and Generate Report".
   - Download the generated PDF (`mca_cmm_report.pdf`) if successful.

## API Endpoints
- **GET /**: Returns a welcome message.
- **POST /extract-info/**: Upload an Excel file to extract metadata and parameter information.
- **POST /generate-pdf/**: Upload an Excel file to generate a PDF report, returned as a base64-encoded string.

## Usage Example
1. Upload an Excel file via the web interface or API.
2. The backend processes the file using `utils.py` to extract metadata and compute statistics.
3. For PDF generation, `pdf_render.py` uses the `empty_reporting.html` template to create a multi-page report with scatter plots.
4. Download the generated PDF from the web interface or API response.

## Dependencies
Key Python packages (listed in `server/requirements.txt`):
- `fastapi`: API server.
- `uvicorn`: ASGI server for FastAPI.
- `pandas`: Excel file processing.
- `numpy`, `scipy`, `statsmodels`: Statistical analysis.
- `matplotlib`: Scatter plot generation.
- `jinja2`, `pdfkit`: PDF rendering.
- `python-multipart`: File uploads.

## Troubleshooting
- **wkhtmltopdf Errors**: Ensure `wkhtmltopdf` is installed and accessible in the system PATH.
- **SSL Errors**: Verify `cert.pem` and `key.pem` are valid and correctly referenced.
- **Excel File Errors**: Ensure the Excel file matches the expected format (see `DI REPORT OF DISC SEPERATOR.xlsx`).
- **API Errors**: Check server logs for detailed error messages.

## Contributing
Contributions are welcome! Submit a pull request or open an issue for bugs, feature requests, or improvements.

## License
See the [LICENSE](./LICENSE) file for details.

## Contact
Developed and maintained by the Database & Analytics Team, C-DAC Chennai.
For support, contact <insert-contact-email>.

© 2025 C-DAC Chennai. All rights reserved.