# Generate CMM Report API Documentation


## API Endpoint

**POST** `/generate-cmm-report/`

## Description
This API endpoint generates a PDF report from JSON inspection data provided in the request body. The report is returned as a base64-encoded string in the response.

## Request

### Headers
- **Content-Type**: `application/json`

### Body
The request body must be a JSON object containing inspection data with the following structure:

#### Required Fields
- **metadata**: Object containing inspection metadata.
- **parameter_info**: Object containing measurement parameter details.

#### Example JSON Body
```json
{
  "metadata": {
    "Inspection_Agency": "CNA (OF), Pune",
    "File_Number": "5752",
    "SO_Number": "AFK SO NO GEMC-511687713116852",
    "SO_Date": "2024-03-07T00:00:00",
    "Firm_Name": "M/s MICRO INDIA ENGINEERING",
    "Offer_Date": "2025-07-05T00:00:00",
    "Qty_on_Order": "2033 Pieces",
    "Qty_Submitted": "953 Pieces",
    "Component": "DISK SEPERATOR",
    "Drawing_Number": "DLG/MCDJ/DISK SEPERATOR/01",
    "Sub_Assembly": "nan",
    "Verified_by": "B N DIVEKAR CTO (MECH)",
    "End_Use_Main_Store": "KAVACH ROCKET",
    "Inspected_by": "MANISH SHARMA GE(MECH)",
    "Operation_Name": "Unknown",
    "Operation_Number": "Unknown",
    "mach_descr": "Unknown",
    "Part_Number": "5752",
    "Part_Description": "DISK SEPERATOR"
  },
  "parameter_info": {
    "O.D": {
      "char_number": "1",
      "char_descr": "O.D",
      "char_class": "Unknown",
      "unit": "Unknown",
      "USL": 89,
      "LSL": 88,
      "Nominal_Value": 88.5,
      "evaluation_starttime": "2025-07-05T00:00:00",
      "evaluation_endtime": "2025-07-05T00:00:00",
      "Subgroup_Size": "Unknown",
      "Subgroup_Type": "Unknown",
      "value_count": 125,
      "Tolerance": 1,
      "measurements": [
        88.71, 88.72, 88.61, 88.56, 88.47, 88.56, 88.75, 88.6, 88.48, 88.49,
        88.57, 88.7, 88.44, 88.62, 88.78, 88.5, 88.62, 88.42, 88.5, 88.61,
        88.64, 88.44, 88.66, 88.4, 88.41, 88.71, 88.58, 88.8, 88.71, 88.49,
        88.5, 88.48, 88.61, 88.64, 88.8, 88.48, 88.52, 88.49, 88.57, 88.51,
        88.41, 88.38, 88.66, 88.67, 88.81, 88.77, 88.38, 88.54, 88.7, 88.57,
        88.8, 88.47, 88.2, 88.41, 88.48, 88.34, 88.74, 88.61, 88.64, 88.4,
        88.48, 88.48, 88.36, 88.74, 88.88, 88.42, 88.48, 88.67, 88.38, 88.67,
        88.5, 88.4, 88.65, 88.57, 88.66, 88.38, 88.59, 88.69, 88.53, 88.8,
        88.52, 88.46, 88.57, 88.72, 88.35, 88.55, 88.48, 88.5, 88.55, 88.33,
        88.48, 88.41, 88.55, 88.36, 88.53, 88.69, 88.72, 88.54, 88.68, 88.78,
        88.54, 88.25, 88.32, 88.47, 88.46, 88.68, 88.46, 88.54, 88.56, 88.78,
        88.55, 88.46, 88.55, 88.4, 88.52, 88.64, 88.62, 88.8, 88.48, 88.71,
        88.41, 88.51, 88.6, 88.62
      ],
      "cp_target": "1.33",
      "cpk_target": "1.33"
    },
    "I.D": {
      "char_number": "2",
      "char_descr": "I.D",
      "char_class": "Unknown",
      "unit": "Unknown",
      "USL": 89,
      "LSL": 88,
      "Nominal_Value": 88.5,
      "evaluation_starttime": "2025-07-05T00:00:00",
      "evaluation_endtime": "2025-07-05T00:00:00",
      "Subgroup_Size": "Unknown",
      "Subgroup_Type": "Unknown",
      "value_count": 125,
      "Tolerance": 1,
      "measurements": [
        88.71, 88.72, 88.61, 88.56, 88.47, 88.56, 88.75, 88.6, 88.48, 88.49,
        88.57, 88.7, 88.44, 88.62, 88.78, 88.5, 88.62, 88.42, 88.5, 88.61,
        88.64, 88.44, 88.66, 88.4, 88.41, 88.71, 88.58, 88.8, 88.71, 88.49,
        88.5, 88.48, 88.61, 88.64, 88.8, 88.48, 88.52, 88.49, 88.57, 88.51,
        88.41, 88.38, 88.66, 88.67, 88.81, 88.77, 88.38, 88.54, 88.7, 88.57,
        88.8, 88.47, 88.2, 88.41, 88.48, 88.34, 88.74, 88.61, 88.64, 88.4,
        88.48, 88.48, 88.36, 88.74, 88.88, 88.42, 88.48, 88.67, 88.38, 88.67,
        88.5, 88.4, 88.65, 88.57, 88.66, 88.38, 88.59, 88.69, 88.53, 88.8,
        88.52, 88.46, 88.57, 88.72, 88.35, 88.55, 88.48, 88.5, 88.55, 88.33,
        88.48, 88.41, 88.55, 88.36, 88.53, 88.69, 88.72, 88.54, 88.68, 88.78,
        88.54, 88.25, 88.32, 88.47, 88.46, 88.68, 88.46, 88.54, 88.56, 88.78,
        88.55, 88.46, 88.55, 88.4, 88.52, 88.64, 88.62, 88.8, 88.48, 88.71,
        88.41, 88.51, 88.6, 88.62
      ],
      "cp_target": "1.33",
      "cpk_target": "1.33"
    }
  }
}
```

#### Metadata Fields
- **Inspection_Agency**: Name of the inspecting agency (string).
- **File_Number**: Unique file identifier (string).
- **SO_Number**: Sales order number (string).
- **SO_Date**: Sales order date (ISO 8601 format).
- **Firm_Name**: Name of the firm (string).
- **Offer_Date**: Offer date (ISO 8601 format).
- **Qty_on_Order**: Quantity ordered (string).
- **Qty_Submitted**: Quantity submitted for inspection (string).
- **Component**: Component name (string).
- **Drawing_Number**: Drawing number (string).
- **Sub_Assembly**: Sub-assembly name (string, can be "nan").
- **Verified_by**: Name of the verifier (string).
- **End_Use_Main_Store**: End-use description (string).
- **Inspected_by**: Name of the inspector (string).
- **Operation_Name**: Operation name (string, can be "Unknown").
- **Operation_Number**: Operation number (string, can be "Unknown").
- **mach_descr**: Machine description (string, can be "Unknown").
- **Part_Number**: Part number (string).
- **Part_Description**: Part description (string).

#### Parameter Info Fields
Each parameter (e.g., "O.D") is an object with the following fields:
- **char_number**: Characteristic number (string).
- **char_descr**: Characteristic description (string).
- **char_class**: Characteristic class (string, can be "Unknown").
- **unit**: Measurement unit (string, can be "Unknown").
- **USL**: Upper specification limit (number).
- **LSL**: Lower specification limit (number).
- **Nominal_Value**: Nominal value (number).
- **evaluation_starttime**: Start time of evaluation (ISO 8601 format).
- **evaluation_endtime**: End time of evaluation (ISO 8601 format).
- **Subgroup_Size**: Subgroup size (string, can be "Unknown").
- **Subgroup_Type**: Subgroup type (string, can be "Unknown").
- **value_count**: Number of measurements (number).
- **Tolerance**: Tolerance value (number).
- **measurements**: Array of measurement values (numbers).
- **cp_target**: Process capability target (string).
- **cpk_target**: Process capability index target (string).

## Response

### Success Response
- **Status Code**: `200 OK`
- **Content-Type**: `application/json`

#### Response Body
```json
{
  "message": "PDF generated successfully.",
  "filename": "mca_cmm_report_json.pdf",
  "content": "<base64-encoded PDF content>"
}
```

#### Fields
- **message**: Success message (string).
- **filename**: Name of the generated PDF file (string).
- **content**: Base64-encoded PDF content (string).

### Error Responses
- **Status Code**: `400 Bad Request`
  - **Example**:
    ```json
    {
      "detail": "Invalid JSON format: 'metadata' and 'parameter_info' keys are required."
    }
    ```
  - **Description**: Returned when the JSON body is invalid or missing required keys.
- **Status Code**: `400 Bad Request`
  - **Example**:
    ```json
    {
      "detail": "Failed to process JSON data or no valid data found."
    }
    ```
  - **Description**: Returned when the JSON data cannot be processed.
- **Status Code**: `500 Internal Server Error`
  - **Example**:
    ```json
    {
      "detail": "Internal server error: <error message>"
    }
    ```
  - **Description**: Returned for unexpected server errors.


## Example Usage
### Curl Command
```bash
curl -X POST "http://<your-api-base-url>/generate-cmm-report/" \
-H "Content-Type: application/json" \
-d '{
  "metadata": {
    "Inspection_Agency": "CNA (OF), Pune",
    "File_Number": "5752",
    "SO_Number": "AFK SO NO GEMC-511687713116852",
    "SO_Date": "2024-03-07T00:00:00",
    "Firm_Name": "M/s MICRO INDIA ENGINEERING",
    "Offer_Date": "2025-07-05T00:00:00",
    "Qty_on_Order": "2033 Pieces",
    "Qty_Submitted": "953 Pieces",
    "Component": "DISK SEPERATOR",
    "Drawing_Number": "DLG/MCDJ/DISK SEPERATOR/01",
    "Sub_Assembly": "nan",
    "Verified_by": "B N DIVEKAR CTO (MECH)",
    "End_Use_Main_Store": "KAVACH ROCKET",
    "Inspected_by": "MANISH SHARMA GE(MECH)",
    "Operation_Name": "Unknown",
    "Operation_Number": "Unknown",
    "mach_descr": "Unknown",
    "Part_Number": "5752",
    "Part_Description": "DISK SEPERATOR"
  },
  "parameter_info": {
    "O.D": {
      "char_number": "1",
      "char_descr": "O.D",
      "char_class": "Unknown",
      "unit": "Unknown",
      "USL": 89,
      "LSL": 88,
      "Nominal_Value": 88.5,
      "evaluation_starttime": "2025-07-05T00:00:00",
      "evaluation_endtime": "2025-07-05T00:00:00",
      "Subgroup_Size": "Unknown",
      "Subgroup_Type": "Unknown",
      "value_count": 125,
      "Tolerance": 1,
      "measurements": [
        88.71, 88.72, 88.61, 88.56, 88.47, 88.56, 88.75, 88.6, 88.48, 88.49,
        88.57, 88.7, 88.44, 88.62, 88.78, 88.5, 88.62, 88.42, 88.5, 88.61,
        88.64, 88.44, 88.66, 88.4, 88.41, 88.71, 88.58, 88.8, 88.71, 88.49,
        88.5, 88.48, 88.61, 88.64, 88.8, 88.48, 88.52, 88.49, 88.57, 88.51,
        88.41, 88.38, 88.66, 88.67, 88.81, 88.77, 88.38, 88.54, 88.7, 88.57,
        88.8, 88.47, 88.2, 88.41, 88.48, 88.34, 88.74, 88.61, 88.64, 88.4,
        88.48, 88.48, 88.36, 88.74, 88.88, 88.42, 88.48, 88.67, 88.38, 88.67,
        88.5, 88.4, 88.65, 88.57, 88.66, 88.38, 88.59, 88.69, 88.53, 88.8,
        88.52, 88.46, 88.57, 88.72, 88.35, 88.55, 88.48, 88.5, 88.55, 88.33,
        88.48, 88.41, 88.55, 88.36, 88.53, 88.69, 88.72, 88.54, 88.68, 88.78,
        88.54, 88.25, 88.32, 88.47, 88.46, 88.68, 88.46, 88.54, 88.56, 88.78,
        88.55, 88.46, 88.55, 88.4, 88.52, 88.64, 88.62, 88.8, 88.48, 88.71,
        88.41, 88.51, 88.6, 88.62
      ],
      "cp_target": "1.33",
      "cpk_target": "1.33"
    },
    "I.D": {
      "char_number": "2",
      "char_descr": "I.D",
      "char_class": "Unknown",
      "unit": "Unknown",
      "USL": 89,
      "LSL": 88,
      "Nominal_Value": 88.5,
      "evaluation_starttime": "2025-07-05T00:00:00",
      "evaluation_endtime": "2025-07-05T00:00:00",
      "Subgroup_Size": "Unknown",
      "Subgroup_Type": "Unknown",
      "value_count": 125,
      "Tolerance": 1,
      "measurements": [
        88.71, 88.72, 88.61, 88.56, 88.47, 88.56, 88.75, 88.6, 88.48, 88.49,
        88.57, 88.7, 88.44, 88.62, 88.78, 88.5, 88.62, 88.42, 88.5, 88.61,
        88.64, 88.44, 88.66, 88.4, 88.41, 88.71, 88.58, 88.8, 88.71, 88.49,
        88.5, 88.48, 88.61, 88.64, 88.8, 88.48, 88.52, 88.49, 88.57, 88.51,
        88.41, 88.38, 88.66, 88.67, 88.81, 88.77, 88.38, 88.54, 88.7, 88.57,
        88.8, 88.47, 88.2, 88.41, 88.48, 88.34, 88.74, 88.61, 88.64, 88.4,
        88.48, 88.48, 88.36, 88.74, 88.88, 88.42, 88.48, 88.67, 88.38, 88.67,
        88.5, 88.4, 88.65, 88.57, 88.66, 88.38, 88.59, 88.69, 88.53, 88.8,
        88.52, 88.46, 88.57, 88.72, 88.35, 88.55, 88.48, 88.5, 88.55, 88.33,
        88.48, 88.41, 88.55, 88.36, 88.53, 88.69, 88.72, 88.54, 88.68, 88.78,
        88.54, 88.25, 88.32, 88.47, 88.46, 88.68, 88.46, 88.54, 88.56, 88.78,
        88.55, 88.46, 88.55, 88.4, 88.52, 88.64, 88.62, 88.8, 88.48, 88.71,
        88.41, 88.51, 88.6, 88.62
      ],
      "cp_target": "1.33",
      "cpk_target": "1.33"
    }
  }
}'
```

## API Endpoint
**POST** `/calculate_graph_statistics/`

## Description
Processes a nested JSON dictionary containing measurement data for multiple parameters and calculates statistical metrics for each parameter. The input includes Upper Specification Limit (USL), Lower Specification Limit (LSL), and a list of measurements. The API appends statistical calculations (mean, standard deviation, max, min, upper control limit, and lower control limit) to each parameter's data and returns the updated dictionary.

## Request

### Curl Command
```bash
curl -X 'POST' \
  'http://localhost:8080/calculate_graph_statistics/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "O.D": {
    "USL": 89,
    "LSL": 88,
    "measurements": [
      88.71, 88.72, 88.61, 88.56, 88.47, 88.56, 88.75, 88.6, 88.48, 88.49,
      88.57, 88.7, 88.44, 88.62, 88.78, 88.5, 88.62, 88.42, 88.5, 88.61,
      88.64, 88.44, 88.66, 88.4, 88.41, 88.71, 88.58, 88.8, 88.71, 88.49,
      88.5, 88.48, 88.61, 88.64, 88.8, 88.48, 88.52, 88.49, 88.57, 88.51
    ]
  },
  "I.D": {
    "USL": 89,
    "LSL": 88,
    "measurements": [
      88.71, 88.72, 88.61, 88.56, 88.47, 88.56, 88.75, 88.6, 88.48, 88.49,
      88.57, 88.7, 88.44, 88.62, 88.78, 88.5, 88.62, 88.42, 88.5, 88.61,
      88.64, 88.44, 88.66, 88.4, 88.41, 88.71, 88.58, 88.8, 88.71, 88.49,
      88.5, 88.48, 88.61, 88.64, 88.8, 88.48, 88.52, 88.49, 88.57, 88.51
    ]
  }
}'
```

### Headers
- **Accept**: `application/json`
- **Content-Type**: `application/json`

### Body
A JSON object where each key is a parameter name (e.g., "O.D", "I.D") and each value is a dictionary containing:
- **USL** (`number`): Upper Specification Limit
- **LSL** (`number`): Lower Specification Limit
- **measurements** (`array[number]`): List of measurement values

## Response

### Status Codes
- **200 OK**: Measurements processed successfully
- **400 Bad Request**: Invalid input format or validation error
- **500 Internal Server Error**: Server-side error during processing

### Response Body
A JSON object containing:
- **message** (`string`): Status message
- **data** (`object`): The input dictionary with appended statistics for each parameter, including:
  - **mean** (`number`): Mean of measurements
  - **std** (`number`): Standard deviation
  - **max** (`number`): Maximum measurement
  - **min** (`number`): Minimum measurement
  - **ucl** (`number`): Upper Control Limit
  - **lcl** (`number`): Lower Control Limit

#### Example Response Body
```json
{
  "message": "Measurements processed successfully.",
  "data": {
    "O.D": {
      "USL": 89,
      "LSL": 88,
      "measurements": [
        88.71, 88.72, 88.61, 88.56, 88.47, 88.56, 88.75, 88.6, 88.48, 88.49,
        88.57, 88.7, 88.44, 88.62, 88.78, 88.5, 88.62, 88.42, 88.5, 88.61,
        88.64, 88.44, 88.66, 88.4, 88.41, 88.71, 88.58, 88.8, 88.71, 88.49,
        88.5, 88.48, 88.61, 88.64, 88.8, 88.48, 88.52, 88.49, 88.57, 88.51
      ],
      "mean": 88.5577,
      "std": 0.1348,
      "max": 88.88,
      "min": 88.2,
      "ucl": 88.5941,
      "lcl": 88.5214
    },
    "I.D": {
      "USL": 89,
      "LSL": 88,
      "measurements": [
        88.71, 88.72, 88.61, 88.56, 88.47, 88.56, 88.75, 88.6, 88.48, 88.49,
        88.57, 88.7, 88.44, 88.62, 88.78, 88.5, 88.62, 88.42, 88.5, 88.61,
        88.64, 88.44, 88.66, 88.4, 88.41, 88.71, 88.58, 88.8, 88.71, 88.49,
        88.5, 88.48, 88.61, 88.64, 88.8, 88.48, 88.52, 88.49, 88.57, 88.51
      ],
      "mean": 88.5577,
      "std": 0.1348,
      "max": 88.88,
      "min": 88.2,
      "ucl": 88.5941,
      "lcl": 88.5214
    }
  }
}
```

## Error Responses

### 400 Bad Request
Returned for invalid input, such as:
- Non-dictionary input
- Empty dictionary
- Missing required keys (`USL`, `LSL`, `measurements`)
- Non-numeric `USL` or `LSL`
- `USL` ≤ `LSL`
- Non-list `measurements`

#### Example Error Response
```json
{
  "detail": "USL must be greater than LSL for 'O.D'"
}
```

### 500 Internal Server Error
Returned for unexpected server-side errors.

#### Example Error Response
```json
{
  "detail": "Internal server error: [error description]"
}
```
