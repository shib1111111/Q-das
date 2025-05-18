from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from utils import analyze_inspection_data
from pdf_render import render_report_to_pdf
import tempfile
import os
import base64
import logging

api_router = APIRouter()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def validate_file(file: UploadFile):
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Only Excel (.xlsx) files are supported.")


@api_router.post("/extract-info/")
async def extract_info(file: UploadFile = File(...)):
    validate_file(file)

    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # Analyze data
        metadata, parameter_info = analyze_inspection_data(temp_path)

        # Clean up Excel file
        os.remove(temp_path)

        return JSONResponse(content={
            "metadata": metadata,
            "parameter_info": parameter_info
        })

    except Exception as e:
        logger.exception("Failed to extract metadata")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@api_router.post("/generate-pdf/")
async def generate_pdf(file: UploadFile = File(...)):
    validate_file(file)

    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # Analyze Excel file
        metadata, parameter_info = analyze_inspection_data(temp_path)

        # Ensure the output directory exists
        output_dir = 'temp_file'
        os.makedirs(output_dir, exist_ok=True)
        output_pdf_path = os.path.join(output_dir, 'mca_cmm_report.pdf')

        # Render PDF
        output_pdf = render_report_to_pdf(metadata, parameter_info, output_pdf=output_pdf_path)

        # Encode PDF as base64
        with open(output_pdf, "rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

        # Clean up temporary files
        os.remove(temp_path)
        os.remove(output_pdf)

        return JSONResponse(content={
            "filename": os.path.basename(output_pdf),
            "content": pdf_base64
        })

    except Exception as e:
        logger.exception("Failed to generate PDF")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
