import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import normal_ad
from typing import List, Dict, Union, Optional
from datetime import datetime
import os
import base64
import io
import matplotlib.pyplot as plt


# Utility Functions
def compute_process_capability(mean, std, usl, lsl, sample_size):
    """Compute Cp and Cpk indices with confidence intervals."""
    try:
        if std == 0 or np.isnan(std):
            return {
                'Cp': float('inf'), 'Cpk': float('inf'),
                'Cp_lower': float('inf'), 'Cp_upper': float('inf'),
                'Cpk_lower': float('inf'), 'Cpk_upper': float('inf')
            }
        cp = (usl - lsl) / (6 * std)
        cpk = min((mean - lsl) / (3 * std), (usl - mean) / (3 * std))
        alpha = 0.05
        chi2_lower = stats.chi2.ppf(1 - alpha/2, sample_size - 1)
        chi2_upper = stats.chi2.ppf(alpha/2, sample_size - 1)
        cp_lower = cp * np.sqrt(chi2_upper / (sample_size - 1))
        cp_upper = cp * np.sqrt(chi2_lower / (sample_size - 1))
        z = stats.norm.ppf(1 - alpha / 2)
        se_cpk = np.sqrt((1 / sample_size) + (cpk ** 2 / (2 * (sample_size - 1))))
        cpk_lower = cpk - z * se_cpk
        cpk_upper = cpk + z * se_cpk
        return {
            'Cp': cp, 'Cpk': cpk,
            'Cp_lower': cp_lower, 'Cp_upper': cp_upper,
            'Cpk_lower': cpk_lower, 'Cpk_upper': cpk_upper
        }
    except Exception as e:
        print(f"Error computing process capability: {e}")
        return {
            'Cp': float('inf'), 'Cpk': float('inf'),
            'Cp_lower': float('inf'), 'Cp_upper': float('inf'),
            'Cpk_lower': float('inf'), 'Cpk_upper': float('inf')
        }

def determine_best_distribution(measurements):
    """Determine the best distribution fit using AIC and Anderson-Darling test."""
    try:
        measurements = np.asarray(measurements)
        if len(measurements) < 3:
            return {'distribution': 'Normal', 'p_value': 1.0, 'method': 'Anderson-Darling'}

        ad_statistic, p_value = normal_ad(measurements)
        method = 'Anderson-Darling'
        if p_value >= 0.05:
            return {'distribution': 'Normal', 'p_value': p_value, 'method': method}

        distributions = [
            ('Normal', stats.norm), ('Log-Normal', stats.lognorm), ('Exponential', stats.expon),
            ('Gamma', stats.gamma), ('Weibull', stats.weibull_min), ('Rayleigh', stats.rayleigh),
            ('Beta', stats.beta)
        ]
        results = []
        min_val = min(measurements)
        data = measurements - min_val + 1e-6 if min_val <= 0 else measurements
        for name, dist in distributions:
            try:
                if name in ['Log-Normal', 'Gamma', 'Weibull', 'Rayleigh']:
                    params = dist.fit(data, floc=0)
                elif name == 'Beta':
                    data_scaled = (data - min(data) + 1e-6) / (max(data) - min(data) + 2e-6)
                    if min(data_scaled) > 0 and max(data_scaled) < 1:
                        params = dist.fit(data_scaled, floc=0, fscale=1)
                    else:
                        continue
                else:
                    params = dist.fit(data)
                log_likelihood = np.sum(dist.logpdf(data, *params))
                aic = 2 * len(params) - 2 * log_likelihood
                results.append((name, aic))
            except Exception:
                continue

        if not results:
            return {'distribution': 'Normal', 'p_value': p_value, 'method': method}
        best_dist = min(results, key=lambda x: x[1])
        return {'distribution': best_dist[0], 'p_value': p_value, 'method': method}
    except Exception as e:
        print(f"Error determining distribution: {e}")
        return {'distribution': 'Normal', 'p_value': 1.0, 'method': 'Anderson-Darling'}

def calculate_statistics(measurements, usl, lsl, cp_target=1.33, cpk_target=1.33):
    """Calculate statistical metrics for measurements."""
    try:
        sample_size = len(measurements)
        mean = np.mean(measurements)
        median = np.median(measurements)
        std = np.std(measurements, ddof=1) if sample_size > 1 else 0
        min_val = np.min(measurements)
        max_val = np.max(measurements)
        range_val = max_val - min_val
        percentiles = {
            'p0_135': np.percentile(measurements, 0.135),
            'p50': np.percentile(measurements, 50),
            'p99_865': np.percentile(measurements, 99.865)
        }
        n_below_lsl = np.sum(np.array(measurements) < lsl)
        n_above_usl = np.sum(np.array(measurements) > usl)
        n_within_tolerance = np.sum((measurements >= lsl) & (measurements <= usl))
        p_below_lsl = (n_below_lsl / sample_size) * 100 if sample_size > 0 else 0
        p_above_usl = (n_above_usl / sample_size) * 100 if sample_size > 0 else 0
        p_within_tolerance = (n_within_tolerance/ sample_size) * 100 if sample_size > 0 else 0
        capability = compute_process_capability(mean, std, usl, lsl, sample_size)
        distribution = determine_best_distribution(measurements)
        cp_requirements_met = capability['Cp'] >= cp_target
        cpk_requirements_met = capability['Cpk'] >= cpk_target
        requirements_met = cp_requirements_met and cpk_requirements_met

        return {
            'sample_size': sample_size, 'effective_sample_size': sample_size,
            'mean': mean, 'median': median, 'std': std,
            'min': min_val, 'max': max_val, 'range': range_val,
            'percentile_0_135': percentiles['p0_135'], 'percentile_50': percentiles['p50'],
            'percentile_99_865': percentiles['p99_865'],
            'n_below_lsl': n_below_lsl, 'n_above_usl': n_above_usl,'n_within_tolerance':n_within_tolerance,
            'p_below_lsl': p_below_lsl, 'p_above_usl': p_above_usl,'p_within_tolerance':p_within_tolerance,
            'Cp': capability['Cp'], 'Cpk': capability['Cpk'],
            'Cp_lower': capability['Cp_lower'], 'Cp_upper': capability['Cp_upper'],
            'Cpk_lower': capability['Cpk_lower'], 'Cpk_upper': capability['Cpk_upper'],
            'distribution_name': distribution['distribution'],
            'distribution_p_value': distribution['p_value'],
            'distribution_method': distribution['method'],
            'cp_requirements_met': cp_requirements_met,
            'cpk_requirements_met': cpk_requirements_met,
            'requirements_met': requirements_met,
            'cp_target': cp_target, 'cpk_target': cpk_target
        }
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        return {
            'sample_size': 0, 'effective_sample_size': 0,
            'mean': 0, 'median': 0, 'std': 0,
            'min': 0, 'max': 0, 'range': 0,
            'percentile_0_135': 0, 'percentile_50': 0, 'percentile_99_865': 0,
            'n_below_lsl': 0, 'n_above_usl': 0,'n_within_tolerance':0,
            'p_below_lsl': 0, 'p_above_usl': 0,'p_within_tolerance':0,
            'Cp': float('inf'), 'Cpk': float('inf'),
            'Cp_lower': float('inf'), 'Cp_upper': float('inf'),
            'Cpk_lower': float('inf'), 'Cpk_upper': float('inf'),
            'distribution_name': 'Unknown',
            'distribution_p_value': 0, 'distribution_method': 'Unknown',
          'cp_requirements_met': False, 'cpk_requirements_met': False,
            'requirements_met': False,
            'cp_target': cp_target, 'cpk_target': cpk_target
        }

def create_scatter_plot(param_name: str, measurements: List[float], stats: Dict[str, float], usl: float, lsl: float, dpi: int = 300) -> Optional[str]:
    try:
        # Input validation
        if not param_name or not isinstance(param_name, str):
            raise ValueError("param_name must be a non-empty string")
        if not measurements or not isinstance(measurements, (list, tuple)):
            raise ValueError("measurements must be a non-empty list or tuple")
        if not all(isinstance(m, (int, float)) for m in measurements):
            raise ValueError("all measurements must be numeric")
        if not stats or not isinstance(stats, dict):
            raise ValueError("stats must be a dictionary")
        if 'mean' not in stats or 'std' not in stats:
            raise KeyError("stats dictionary must contain 'mean' and 'std' keys")
        if not isinstance(usl, (int, float)) or not isinstance(lsl, (int, float)):
            raise ValueError("usl and lsl must be numeric")
        if lsl >= usl:
            raise ValueError("lsl must be less than usl")
        if stats['std'] < 0:
            raise ValueError("standard deviation cannot be negative")
        if dpi < 72 or dpi > 1200:  # Reasonable DPI bounds
            raise ValueError("DPI must be between 72 and 1200")

        # Set up figure
        plt.figure(figsize=(15, 6), dpi=dpi)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Prepare data
        x = np.arange(1, len(measurements) + 1)
        measurements = np.array(measurements)

        # Define ranges for coloring points
        ranges = [
            (lsl, stats['mean'] - 3 * stats['std'], 'red'),
            (stats['mean'] - 3 * stats['std'], stats['mean'] + 3 * stats['std'], 'green'),
            (stats['mean'] + 3 * stats['std'], usl, 'blue')
        ]
        out_of_spec_color = 'purple'

        # Assign colors
        colors = []
        for m in measurements:
            if m < lsl or m > usl:
                colors.append(out_of_spec_color)
            else:
                for lower, upper, color in ranges:
                    if lower <= m <= upper:
                        colors.append(color)
                        break

        # Plot scatter points without legend
        for x_val, y_val, color in zip(x, measurements, colors):
            plt.scatter(x_val, y_val, c=color, s=50, zorder=5)

        # Add horizontal lines
        line_configs = [
            ('USL', usl, 'purple', ':', 'USL'),
            ('x̄ + 3s', stats['mean'] + 3 * stats['std'], 'orange', '--', 'x̄ + 3s'),
            ('x̄', stats['mean'], 'blue', '-.', 'x̄'),
            ('x̄ - 3s', stats['mean'] - 3 * stats['std'], 'orange', '--', 'x̄ - 3s'),
            ('LSL', lsl, 'purple', ':', 'LSL')
        ]

        for _, y_value, color, linestyle, label in line_configs:
            plt.axhline(y=y_value, color=color, linestyle=linestyle, linewidth=2, zorder=3)
            plt.text(len(measurements) + 0.2, y_value, label, color=color, va='center', fontsize=10)

        # Calculate y-axis range
        y_range_padding = 0.1 * (usl - lsl)
        y_min = min(lsl, measurements.min()) - y_range_padding
        y_max = max(usl, measurements.max()) + y_range_padding

        # Customize plot
        plt.title(f'Value Chart for {param_name}', fontsize=14, pad=10)
        plt.xlabel('Value No.', fontsize=12)
        plt.ylabel(param_name, fontsize=12)
        plt.xlim(0.5, len(measurements) + 1.5)
        plt.ylim(y_min, y_max)

        # Generate base64-encoded PNG
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=False)
        plt.close()
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"

    except ValueError as ve:
        print(f"Validation error in create_scatter_plot for '{param_name}': {ve}")
        return None
    except KeyError as ke:
        print(f"Key error in create_scatter_plot for '{param_name}': {ke}")
        return None
    except Exception as e:
        print(f"Unexpected error in create_scatter_plot for '{param_name}': {e}")
        return None
    
def parse_metadata(df):
    """Parse metadata from the Excel dataframe."""
    try:
        return {
            'Inspection_Agency': df.iloc[0, 1] if df.shape[0] > 0 and df.shape[1] > 1 else 'Unknown',
            'File_Number': df.iloc[0, 3] if df.shape[0] > 0 and df.shape[1] > 3 else 'Unknown',
            'SO_Number': df.iloc[1, 1] if df.shape[0] > 1 and df.shape[1] > 1 else 'Unknown',
            'SO_Date': df.iloc[1, 3] if df.shape[0] > 1 and df.shape[1] > 3 else 'Unknown',
            'Firm_Name': df.iloc[2, 1] if df.shape[0] > 2 and df.shape[1] > 1 else 'Unknown',
            'Offer_Date': df.iloc[2, 3] if df.shape[0] > 2 and df.shape[1] > 3 else 'Unknown',
            'Qty_on_Order': df.iloc[3, 1] if df.shape[0] > 3 and df.shape[1] > 1 else 'Unknown',
            'Qty_Submitted': df.iloc[3, 3] if df.shape[0] > 3 and df.shape[1] > 3 else 'Unknown',
            'Component': df.iloc[4, 1] if df.shape[0] > 4 and df.shape[1] > 1 else 'Unknown',
            'Drawing_Number': df.iloc[4, 3] if df.shape[0] > 4 and df.shape[1] > 3 else 'Unknown',
            'Sub_Assembly': df.iloc[5, 1] if df.shape[0] > 5 and df.shape[1] > 1 else 'Unknown',
            'Verified_by': df.iloc[5, 3] if df.shape[0] > 5 and df.shape[1] > 3 else 'Unknown',
            'End_Use_Main_Store': df.iloc[6, 1] if df.shape[0] > 6 and df.shape[1] > 1 else 'Unknown',
            'Inspected_by': df.iloc[6, 3] if df.shape[0] > 6 and df.shape[1] > 3 else 'Unknown',
            'Operation_Name': 'Unknown',
            'Operation_Number': 'Unknown',
            'mach_descr': 'Unknown',
            'Part_Number': df.iloc[0, 3] if df.shape[0] > 0 and df.shape[1] > 3 else 'Unknown',
            'Part_Description': df.iloc[4, 1] if df.shape[0] > 4 and df.shape[1] > 1 else 'Unknown',
        }
    except Exception as e:
        print(f"Error parsing metadata: {e}")
        return {}

def analyze_inspection_data(file_path):
    """Analyze Excel file and return metadata and parameter information."""
    try:
        df = pd.read_excel(file_path, header=1)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None, {}


    metadata = parse_metadata(df)

    try:
        param_start_row = df[df.iloc[:, 0].astype(str).str.contains('Parameters', na=False)].index[0]
    except IndexError:
        print("Could not locate 'Parameters' row in the Excel file.")
        return metadata, {}

    try:
        df_params = pd.read_excel(file_path, skiprows=param_start_row + 1, header=1)
        columns = df_params.columns
        param_metadata_rows = df_params.iloc[:3].reset_index(drop=True)
        measurement_data = df_params.iloc[3:].reset_index(drop=True)
    except Exception as e:
        print(f"Error reading parameter section: {e}")
        return metadata, {}

    parameter_info = {}
    for i in range(2, len(columns), 2):
        param_name = columns[i]
        date_col = columns[i - 1]
        try:
            values = measurement_data[param_name]
            dates = measurement_data[date_col]
            valid_values = values.dropna()
            valid_dates = dates.loc[valid_values.index].dropna()

            usl = param_metadata_rows.iloc[0, i]
            lsl = param_metadata_rows.iloc[1, i]
            nominal = param_metadata_rows.iloc[2, i]
            tolerance = usl - lsl if pd.notna(usl) and pd.notna(lsl) else 'Unknown'

            param_data = {
                'char_number': i // 2,
                'char_descr': param_name,
                'char_class': 'Unknown',
                'unit': 'Unknown',
                'USL': usl,
                'LSL': lsl,
                'Nominal_Value': nominal,
                'evaluation_starttime': valid_dates.min() if not valid_dates.empty else None,
                'evaluation_endtime': valid_dates.max() if not valid_dates.empty else None,
                'Subgroup_Size': 'Unknown',
                'Subgroup_Type': 'Unknown',
                'value_count': valid_values.count(),
                'Tolerance': tolerance,
                'measurements': valid_values.tolist()
            }

            stats = calculate_statistics(param_data['measurements'], usl, lsl)
            param_data.update(stats)
            param_data['plot_image'] = create_scatter_plot(param_name, param_data['measurements'], stats, usl, lsl)
            parameter_info[param_name] = param_data

        except Exception as e:
            print(f"Error processing parameter '{param_name}': {e}")
            continue

    return metadata, parameter_info

# # Example Usage
# if __name__ == "__main__":
#     try:
#         metadata, parameter_info = analyze_inspection_data('C:\Users\shib kumar saraf\Downloads\Q-das\server\app_utils\DI REPORT OF DISC SEPERATOR.xlsx')
#         print("Metadata:", metadata)
#         for param, info in parameter_info.items():
#             print(f"\nParameter: {param}")
#             for key, value in info.items():
#                 print(f"{key}: {value}")
#     except Exception as e:
#         print(f"Error in main execution: {e}")