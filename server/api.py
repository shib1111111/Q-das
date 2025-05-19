# /server/api.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from app_utils.utils import analyze_inspection_data
from app_utils.pdf_render import render_report_to_pdf
import tempfile
import os
import base64
import logging
import numpy as np
import datetime
import time
from functools import wraps

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    elif hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    try:
        return str(obj)
    except Exception:
        return None

api_router = APIRouter()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Decorator to measure and log time taken
def log_time_taken(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"API call '{func.__name__}' took {elapsed_time:.4f} seconds")
        print(f"API call '{func.__name__}' took {elapsed_time:.4f} seconds")  # Also print to console
        return result
    return wrapper

def validate_file(file: UploadFile):
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Only Excel (.xlsx) files are supported.")

@api_router.post("/extract-info/")
@log_time_taken
async def extract_info(file: UploadFile = File(...)):
    validate_file(file)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        metadata, parameter_info = analyze_inspection_data(temp_path)
        os.remove(temp_path)

        return JSONResponse(content=jsonable_encoder({
            "metadata": sanitize_for_json(metadata),
            "parameter_info": sanitize_for_json(parameter_info)
        }))

    except Exception as e:
        logger.exception("Failed to extract metadata")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api_router.post("/generate-pdf/")
@log_time_taken
async def generate_pdf(file: UploadFile = File(...)):
    validate_file(file)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        metadata, parameter_info = analyze_inspection_data(temp_path)
        output_dir = 'temp_file'
        os.makedirs(output_dir, exist_ok=True)
        output_pdf_path = os.path.join(output_dir, 'mca_cmm_report.pdf')
        output_pdf = render_report_to_pdf(metadata, parameter_info, output_pdf=output_pdf_path)

        with open(output_pdf, "rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

        os.remove(temp_path)
        # os.remove(output_pdf)  # Commented out in original code, kept as is

        return JSONResponse(content={
            "filename": os.path.basename(output_pdf),
            "content": pdf_base64
        })

    except Exception as e:
        logger.exception("Failed to generate PDF")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")