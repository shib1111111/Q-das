# Q-das Production Deployment Guide

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-darkblue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-red)
![wkhtmltopdf](https://img.shields.io/badge/wkhtmltopdf-0.12.6-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Q-das is a web-based application for generating MCA/CMM reports from Excel (.xlsx) files. It processes inspection data, performs statistical analysis (e.g., Cp, Cpk, best-fit distributions), and produces multi-page PDF reports with parameter details, metrics, and scatter plots. The backend uses FastAPI, served via Gunicorn, with Nginx as a reverse proxy, secured by SSL on an Ubuntu remote server.

## Project Structure

```
Q-das/
├── setup_env.sh                  # Script to set up virtual environment and dependencies
├── client/                       # Frontend static files
│   └── index.html                # Main HTML file for web interface
├── server/                       # Backend FastAPI application
│   ├── api.py                    # API endpoints for metadata and PDF generation
│   ├── app.py                    # FastAPI application entry point
│   ├── requirements.txt          # Python dependencies
│   ├── app_utils/                # Utility modules and assets
│   │   ├── pdf_render.py         # PDF rendering using Jinja2 and pdfkit
│   │   ├── utils.py              # Statistical analysis and plot generation
│   │   ├── static/               # Static assets
│   │   │   └── cdac_logo.png     # Logo for reports
│   │   ├── templates/            # Jinja2 templates
│   │   │   └── empty_reporting.html  # Template for PDF reports
│   ├── ca_certificates/          # SSL certificates
│   │   ├── cert.pem              # SSL certificate
│   │   ├── key.pem               # SSL private key
│   ├── temp_file/                # Directory for temporary files and PDFs
```

## Prerequisites

- Ubuntu server (20.04 or later) with root or sudo access.
- Python 3.10+ (`sudo apt install python3 python3-venv`).
- Nginx (`sudo apt install nginx`).
- wkhtmltopdf 0.12.6 (`sudo apt install wkhtmltopdf`).
- Git (optional, `sudo apt install git`).
- Domain name or public IP for production.
- Ports 80 (HTTP) and 443 (HTTPS) open in the firewall.
- SSL certificates in `server/ca_certificates/` or a domain for Let's Encrypt.

## Deployment Instructions

### 1. Clone the Repository
Clone or transfer the Q-das project to the server:
```bash
git clone https://github.com/shib1111111/Q-das.git /home/your_user/Q-das
cd /home/your_user/Q-das
```
Alternatively, use `scp` to copy files to `/home/your_user/Q-das`.

### 2. Set Up the Environment
Run the setup script to create a virtual environment and install dependencies:
```bash
chmod +x setup_env.sh
./setup_env.sh
```
This creates `venv/` and installs dependencies from `server/requirements.txt` (e.g., FastAPI, pandas, matplotlib, pdfkit).

If the script fails, manually set up:
```bash
cd /home/your_user/Q-das/server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Install wkhtmltopdf
Ensure wkhtmltopdf is installed for PDF generation:
```bash
sudo apt update
sudo apt install wkhtmltopdf
```
Verify installation:
```bash
wkhtmltopdf --version
```

### 4. Test the Backend
Verify the FastAPI backend with Gunicorn:
```bash
cd /home/your_user/Q-das/server
source venv/bin/activate
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 127.0.0.1:8000
```
Access `http://127.0.0.1:8000` (returns `{"message": "Welcome to the MCA/CMM Report Generator API"}`) or `http://127.0.0.1:8000/docs` for Swagger UI. Stop with `Ctrl+C`.

### 5. Configure Gunicorn as a Systemd Service
Create a systemd service for Gunicorn:
```bash
sudo nano /etc/systemd/system/q-das.service
```
Add (replace `your_user`):
```
[Unit]
Description=Gunicorn instance for Q-das API
After=network.target

[Service]
User=your_user
Group=www-data
WorkingDirectory=/home/your_user/Q-das/server
Environment="PATH=/home/your_user/Q-das/server/venv/bin"
ExecStart=/home/your_user/Q-das/server/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 127.0.0.1:8000

[Install]
WantedBy=multi-user.target
```
Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable q-das.service
sudo systemctl start q-das.service
```
Verify:
```bash
sudo systemctl status q-das.service
```

### 6. Configure Nginx as a Reverse Proxy
Create an Nginx configuration:
```bash
sudo nano /etc/nginx/sites-available/q-das
```
Add (replace `your_domain_or_ip` and paths):
```
server {
    listen 80;
    listen 443 ssl;
    server_name your_domain_or_ip;

    ssl_certificate /home/your_user/Q-das/server/ca_certificates/cert.pem;
    ssl_certificate_key /home/your_user/Q-das/server/ca_certificates/key.pem;

    location / {
        root /home/your_user/Q-das/client;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
Enable and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/q-das /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 7. Secure with Let's Encrypt SSL (Optional)
For a trusted SSL certificate:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your_domain -d www.your_domain
```
Certbot updates Nginx and enables auto-renewal. Restart Nginx:
```bash
sudo systemctl restart nginx
```

### 8. Configure Firewall
Allow HTTP/HTTPS traffic:
```bash
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
sudo ufw status
```

## Running the Application
- **Frontend**: Access at `https://your_domain_or_ip` to upload Excel files and download PDF reports.
- **API**: Use `https://your_domain_or_ip/api/` or `https://your_domain_or_ip/docs` (Swagger UI) for programmatic access.
- **Functionality**: Upload `.xlsx` files to extract metadata, compute statistics (Cp, Cpk, mean, etc.), and generate PDF reports with scatter plots using `pdf_render.py` and `empty_reporting.html`.

## API Endpoints
- **GET /**: Welcome message.
- **POST /extract-info/**: Upload Excel file to extract metadata and parameters.
- **POST /generate-pdf/**: Upload Excel file to generate a base64-encoded PDF report.

## Troubleshooting
- **502 Bad Gateway**: Verify Gunicorn (`sudo systemctl status q-das.service`) and `proxy_pass` (`http://127.0.0.1:8000`).
- **wkhtmltopdf Errors**: Ensure wkhtmltopdf is installed and in PATH (`wkhtmltopdf --version`).
- **SSL Errors**: Check certificate paths/permissions (`sudo chmod 600 /path/to/cert.pem`).
- **CORS Issues**: Update `allow_origins` in `server/app.py` (e.g., `["https://your_domain"]`).
- **Excel File Errors**: Ensure files match expected format (e.g., `DI REPORT OF DISC SEPERATOR.xlsx`).
- **Logs**:
  - Gunicorn: `journalctl -u q-das.service`
  - Nginx: `sudo tail -f /var/log/nginx/error.log` or `access.log`

## Maintenance
- **Update Dependencies**: `source venv/bin/activate && pip install -r requirements.txt --upgrade`
- **Monitor Logs**: Regularly check Gunicorn/Nginx logs.
- **Backup Certificates**: Secure `server/ca_certificates/` or rely on Let's Encrypt renewal.
- **Scaling**: Adjust Gunicorn workers (`-w`) based on CPU cores (e.g., `2*CPU_cores + 1`).

## License
MIT License (see [LICENSE](./LICENSE) file).

## Contact
Developed by the Database & Analytics Team, C-DAC Chennai. For support, contact <insert-contact-email>.

© 2025 C-DAC Chennai. All rights reserved.