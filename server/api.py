# /server/api.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Dict, Any
# from app_utils.utils import analyze_inspection_data
from app_utils.utils_V2 import analyze_inspection_data,analyze_inspection_data_json,calculate_graph_statistics
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
        os.remove(output_pdf)  # Commented out in original code, kept as is

        return JSONResponse(content={
            "filename": os.path.basename(output_pdf),
            "content": pdf_base64
        })

    except Exception as e:
        logger.exception("Failed to generate PDF")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    

@api_router.post("/generate-cmm-report/")
@log_time_taken
async def generate_cmm_report(inspection_data: Dict[str, Any]):
    """Generate a PDF report from JSON data provided in the request body."""
    try:
        # Basic validation for required keys
        if not isinstance(inspection_data, dict) or "metadata" not in inspection_data or "parameter_info" not in inspection_data:
            raise HTTPException(status_code=400, detail="Invalid JSON format: 'metadata' and 'parameter_info' keys are required.")

        # Analyze JSON data directly from the request body
        metadata, parameter_info = analyze_inspection_data_json(inspection_data)
        
        if not metadata or not parameter_info:
            raise HTTPException(status_code=400, detail="Failed to process JSON data or no valid data found.")

        # Generate PDF
        output_dir = 'temp_file'
        os.makedirs(output_dir, exist_ok=True)
        output_pdf_path = os.path.join(output_dir, 'mca_cmm_report_json.pdf')
        output_pdf = render_report_to_pdf(metadata, parameter_info, output_pdf=output_pdf_path)

        # Read PDF and encode to base64
        with open(output_pdf, "rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

        # Clean up temporary file
        os.remove(output_pdf)

        return JSONResponse(content={
            'message': 'PDF generated successfully.',
            "filename": os.path.basename(output_pdf),
            "content": pdf_base64
        })

    except Exception as e:
        logger.exception("Failed to generate PDF from JSON body")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    


@api_router.post("/calculate_graph_statistics/")
async def graph_statistics(inspection_data: Dict[str, Any]):
    """
    Process nested dictionary JSON data and append statistical calculations.
    Expects a nested dictionary with parameter names as keys and dictionaries containing
    USL, LSL, and measurements as values.
    Returns the input dictionary with appended statistics for each parameter.
    """
    try:
        # Validate input is a dictionary
        if not isinstance(inspection_data, dict):
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON format: Input must be a dictionary"
            )
        
        # Validate dictionary is not empty
        if not inspection_data:
            raise HTTPException(
                status_code=400,
                detail="Input dictionary cannot be empty"
            )
        
        # Process each parameter
        for param_name, param_data in inspection_data.items():
            # Validate parameter data is a dictionary
            if not isinstance(param_data, dict):
                raise HTTPException(
                    status_code=400,
                    detail=f"Data for parameter '{param_name}' must be a dictionary"
                )
            
            # Validate required keys
            required_keys = ["USL", "LSL", "measurements"]
            if not all(key in param_data for key in required_keys):
                raise HTTPException(
                    status_code=400,
                    detail=f"Parameter '{param_name}' missing required keys: {required_keys}"
                )
            
            # Validate USL and LSL are numeric
            if not isinstance(param_data["USL"], (int, float)) or not isinstance(param_data["LSL"], (int, float)):
                raise HTTPException(
                    status_code=400,
                    detail=f"USL and LSL for '{param_name}' must be numeric"
                )
            
            # Validate USL > LSL
            if param_data["USL"] <= param_data["LSL"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"USL must be greater than LSL for '{param_name}'"
                )
            
            # Validate measurements is a list
            if not isinstance(param_data["measurements"], list):
                raise HTTPException(
                    status_code=400,
                    detail=f"Measurements for '{param_name}' must be a list"
                )
            
            # Calculate statistics and update parameter dictionary
            stats = calculate_graph_statistics(param_data["measurements"])
            param_data.update(stats)
        
        return {
            "message": "Measurements processed successfully.",
            "data": inspection_data
        }
    
    except ValueError as ve:
        logger.exception("Validation error in processing measurements")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(ve)}")
    except Exception as e:
        logger.exception("Failed to process measurements")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
