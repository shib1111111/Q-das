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
import json
from math import sqrt
from scipy.stats import norm


# Utility Functions

def get_percentiles(measurements: List[float], distribution: Dict[str, Union[str, float, tuple]]) -> Dict[str, float]:
    dist_name = distribution['distribution']
    params = distribution['params']
    
    # Calculate percentiles based on distribution
    percentiles = {}
    min_val = min(measurements)
    data = measurements - min_val + 1e-6 if min_val <= 0 else measurements  # Shift for Log-Normal, etc.
    
    if dist_name == 'Normal' and params is not None:
        mean, std = params
        percentiles['p0_135'] = stats.norm.ppf(0.00135, loc=mean, scale=std)
        percentiles['p50'] = stats.norm.ppf(0.50, loc=mean, scale=std)
        percentiles['p99_865'] = stats.norm.ppf(0.99865, loc=mean, scale=std)
    
    elif dist_name == 'Log-Normal' and params is not None:
        shape, loc, scale = params
        percentiles['p0_135'] = stats.lognorm.ppf(0.00135, shape, loc=loc, scale=scale)
        percentiles['p50'] = stats.lognorm.ppf(0.50, shape, loc=loc, scale=scale)
        percentiles['p99_865'] = stats.lognorm.ppf(0.99865, shape, loc=loc, scale=scale)
    
    elif dist_name == 'Exponential' and params is not None:
        loc, scale = params
        percentiles['p0_135'] = stats.expon.ppf(0.00135, loc=loc, scale=scale)
        percentiles['p50'] = stats.expon.ppf(0.50, loc=loc, scale=scale)
        percentiles['p99_865'] = stats.expon.ppf(0.99865, loc=loc, scale=scale)
    
    elif dist_name == 'Gamma' and params is not None:
        a, loc, scale = params
        percentiles['p0_135'] = stats.gamma.ppf(0.00135, a, loc=loc, scale=scale)
        percentiles['p50'] = stats.gamma.ppf(0.50, a, loc=loc, scale=scale)
        percentiles['p99_865'] = stats.gamma.ppf(0.99865, a, loc=loc, scale=scale)
    
    elif dist_name == 'Weibull' and params is not None:
        shape, loc, scale = params
        percentiles['p0_135'] = stats.weibull_min.ppf(0.00135, shape, loc=loc, scale=scale)
        percentiles['p50'] = stats.weibull_min.ppf(0.50, shape, loc=loc, scale=scale)
        percentiles['p99_865'] = stats.weibull_min.ppf(0.99865, shape, loc=loc, scale=scale)
    
    elif dist_name == 'Rayleigh' and params is not None:
        loc, scale = params
        percentiles['p0_135'] = stats.rayleigh.ppf(0.00135, loc=loc, scale=scale)
        percentiles['p50'] = stats.rayleigh.ppf(0.50, loc=loc, scale=scale)
        percentiles['p99_865'] = stats.rayleigh.ppf(0.99865, loc=loc, scale=scale)
    
    elif dist_name == 'Beta' and params is not None:
        a, b, loc, scale = params
        # Beta distribution is defined on [0,1], so we need to scale back
        p_0135_scaled = stats.beta.ppf(0.00135, a, b)
        p_50_scaled = stats.beta.ppf(0.50, a, b)
        p_99865_scaled = stats.beta.ppf(0.99865, a, b)
        # Transform back to original scale
        data_range = max(data) - min(data) + 2e-6
        percentiles['p0_135'] = p_0135_scaled * data_range + min(data) - 1e-6
        percentiles['p50'] = p_50_scaled * data_range + min(data) - 1e-6
        percentiles['p99_865'] = p_99865_scaled * data_range + min(data) - 1e-6
    
    else:
        # Fallback to empirical percentiles
        percentiles['p0_135'] = np.percentile(measurements, 0.135)
        percentiles['p50'] = np.percentile(measurements, 50)
        percentiles['p99_865'] = np.percentile(measurements, 99.865)
    
    return percentiles


def compute_process_capability(mean: float, std: float, usl: float, lsl: float, sample_size: int, 
                             distribution_name: str, p0_135: float, p50: float, p99_865: float) -> Dict[str, float]:
    """Compute Cp and Cpk indices with confidence intervals, adjusted for distribution type."""
    try:
        if std == 0 or np.isnan(std) or sample_size < 2:
            return {
                'Cp': float('inf'), 'Cpk': float('inf'),
                'Cp_lower': float('inf'), 'Cp_upper': float('inf'),
                'Cpk_lower': float('inf'), 'Cpk_upper': float('inf')
            }

        if distribution_name == 'Normal':
            cp = (usl - lsl) / (6 * std)
            cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
        else:
            if p99_865 == p0_135:  # Avoid division by zero
                return {
                    'Cp': float('inf'), 'Cpk': float('inf'),
                    'Cp_lower': float('inf'), 'Cp_upper': float('inf'),
                    'Cpk_lower': float('inf'), 'Cpk_upper': float('inf')
                }
            cp = (usl - lsl) / (p99_865 - p0_135)
            cpk = min(((usl - p50) / (p99_865 - p50)), ((p50 - lsl) / (p50 - p0_135)))

        # Confidence intervals for Cp and Cpk
        alpha = 0.05
        chi2_lower = stats.chi2.ppf(1 - alpha / 2, sample_size - 1)
        chi2_upper = stats.chi2.ppf(alpha / 2, sample_size - 1)
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

def determine_best_distribution(measurements: List[float]) -> Dict[str, Union[str, float, tuple]]:
    """Determine the best distribution fit using AIC and Anderson-Darling test."""
    try:
        measurements = np.asarray(measurements)
        if len(measurements) < 3:
            return {'distribution': 'Normal', 'p_value': 1.0, 'method': 'Anderson-Darling', 'params': None}

        ad_statistic, p_value = normal_ad(measurements)
        method = 'Anderson-Darling'
        if p_value >= 0.05:
            return {'distribution': 'Normal', 'p_value': p_value, 'method': method, 'params': (np.mean(measurements), np.std(measurements, ddof=1))}

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
                results.append((name, aic, params))
            except Exception:
                continue

        if not results:
            return {'distribution': 'Normal', 'p_value': p_value, 'method': method, 'params': (np.mean(measurements), np.std(measurements, ddof=1))}
        best_dist = min(results, key=lambda x: x[1])
        return {
            'distribution': best_dist[0],
            'p_value': p_value,
            'method': method,
            'params': best_dist[2]
        }
    except Exception as e:
        print(f"Error determining distribution: {e}")
        return {'distribution': 'Normal', 'p_value': 1.0, 'method': 'Anderson-Darling', 'params': None}

def calculate_statistics(measurements: List[float], usl: float, lsl: float, 
                        cp_target: float = 1.33, cpk_target: float = 1.33, 
                        target: float = None) -> Dict[str, Union[float, int, bool, str]]:
    """Calculate statistical metrics for measurements, including CDF-based probabilities."""
    try:
        sample_size = len(measurements)
        if sample_size == 0:
            return {
                'sample_size': 0, 'effective_sample_size': 0, 'mean': 0, 'median': 0, 'std': 0,
                'min': 0, 'max': 0, 'range': 0, 'percentile_0_135': 0, 'percentile_50': 0,
                'percentile_99_865': 0, 'n_below_lsl': 0, 'n_above_usl': 0, 'n_within_tolerance': 0,
                'p_below_target': 0, 'p_above_usl': 0, 'p_below_lsl': 0,
                'Cp': float('inf'), 'Cpk': float('inf'), 'Cp_lower': float('inf'), 'Cp_upper': float('inf'),
                'Cpk_lower': float('inf'), 'Cpk_upper': float('inf'), 'distribution_name': 'Unknown',
                'distribution_p_value': 0, 'distribution_method': 'Unknown', 'cp_requirements_met': False,
                'cpk_requirements_met': False, 'requirements_met': False, 'cp_target': cp_target,
                'cpk_target': cpk_target
            }

        mean = np.mean(measurements)
        median = np.median(measurements)
        std = np.std(measurements, ddof=1) if sample_size > 1 else 0
        min_val = np.min(measurements)
        max_val = np.max(measurements)
        range_val = max_val - min_val
        percentiles = {
            'p0_135': norm.ppf(0.00135, loc=mean, scale=std),
            'p50': norm.ppf(0.50, loc=mean, scale=std),
            'p99_865': norm.ppf(0.99865, loc=mean, scale=std)
        
            # 'p0_135': np.percentile(measurements, 0.135),
            # 'p50': np.percentile(measurements, 50),
            # 'p99_865': np.percentile(measurements, 99.865)
        }

        n_below_lsl = np.sum(np.array(measurements) < lsl)
        n_above_usl = np.sum(np.array(measurements) > usl)
        n_within_tolerance = np.sum((np.array(measurements) >= lsl) & (np.array(measurements) <= usl))

        distribution = determine_best_distribution(measurements)
        percentiles = get_percentiles(measurements, distribution)
        capability = compute_process_capability(
            mean, std, usl, lsl, sample_size, 
            distribution['distribution'], percentiles['p0_135'], percentiles['p50'], percentiles['p99_865']
        )

        # Calculate probabilities using CDF, with tolerance (USL - LSL) as default target
        tolerance = usl - lsl if pd.notna(usl) and pd.notna(lsl) else 0
        target = tolerance if target is None else target  # Use tolerance as default target
        measurements = np.asarray(measurements)
        params = distribution.get('params')

        # Initialize probabilities
        p_within_tolerance = 0.0
        p_above_usl = 0.0
        p_below_lsl = 0.0

        if params is not None and len(measurements) > 0:
            # Map distribution name to scipy.stats distribution
            dist_map = {
                'Normal': stats.norm, 'Log-Normal': stats.lognorm, 'Exponential': stats.expon,
                'Gamma': stats.gamma, 'Weibull': stats.weibull_min, 'Rayleigh': stats.rayleigh,
                'Beta': stats.beta
            }
            dist = dist_map.get(distribution['distribution'], stats.norm)

            # Calculate CDF values
            if distribution['distribution'] == 'Normal':
                loc, scale = params  # mean, std
                p_below_usl = dist.cdf(usl, loc=loc, scale=scale)
                p_below_lsl = dist.cdf(lsl, loc=loc, scale=scale)
                p_within_tolerance = p_below_usl - p_below_lsl
                p_above_usl = 1 - p_below_usl
            elif distribution['distribution'] == 'Beta':
                a, b, loc, scale = params
                p_below_usl = dist.cdf(usl, a, b, loc=loc, scale=scale)
                p_below_lsl = dist.cdf(lsl, a, b, loc=loc, scale=scale)
                p_within_tolerance = p_below_usl - p_below_lsl
                p_above_usl = 1 - p_below_usl
            else:
                # For other distributions, adjust data if shifted
                min_val = min(measurements)
                shift = min_val - 1e-6 if min_val <= 0 else 0
                p_below_usl = dist.cdf(usl - shift, *params)
                p_below_lsl = dist.cdf(lsl - shift, *params)
                p_within_tolerance = p_below_usl - p_below_lsl
                p_above_usl = 1 - p_below_usl

        # Convert probabilities to percentage
        p_within_tolerance *= 100
        p_above_usl *= 100
        p_below_lsl *= 100
        cp_requirements_met = capability['Cp'] >= cp_target
        cpk_requirements_met = capability['Cpk'] >= cpk_target
        requirements_met = cp_requirements_met and cpk_requirements_met

        return {
            'sample_size': sample_size,
            'effective_sample_size': sample_size,
            'mean': mean,
            'median': median,
            'std': std,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'percentile_0_135': percentiles['p0_135'],
            'percentile_50': percentiles['p50'],
            'percentile_99_865': percentiles['p99_865'],
            'n_below_lsl': n_below_lsl,
            'n_above_usl': n_above_usl,
            'n_within_tolerance': n_within_tolerance,
            'p_below_lsl': p_below_lsl,  # From CDF
            'p_above_usl': p_above_usl,  # From CDF
            'p_within_tolerance': p_within_tolerance,  # P(X < T) where T is tolerance
            'Cp': capability['Cp'],
            'Cpk': capability['Cpk'],
            'Cp_lower': capability['Cp_lower'],
            'Cp_upper': capability['Cp_upper'],
            'Cpk_lower': capability['Cpk_lower'],
            'Cpk_upper': capability['Cpk_upper'],
            'distribution_name': distribution['distribution'],
            'distribution_p_value': distribution['p_value'],
            'distribution_method': distribution['method'],
            'cp_requirements_met': cp_requirements_met,
            'cpk_requirements_met': cpk_requirements_met,
            'requirements_met': requirements_met,
            'cp_target': cp_target,
            'cpk_target': cpk_target
        }
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        return {
            'sample_size': 0, 'effective_sample_size': 0, 'mean': 0, 'median': 0, 'std': 0,
            'min': 0, 'max': 0, 'range': 0, 'percentile_0_135': 0, 'percentile_50': 0,
            'percentile_99_865': 0, 'n_below_lsl': 0, 'n_above_usl': 0, 'n_within_tolerance': 0,
            'p_below_lsl': 0, 'p_above_usl': 0, 'p_within_tolerance': 0,
            'Cp': float('inf'), 'Cpk': float('inf'), 'Cp_lower': float('inf'), 'Cp_upper': float('inf'),
            'Cpk_lower': float('inf'), 'Cpk_upper': float('inf'), 'distribution_name': 'Unknown',
            'distribution_p_value': 0, 'distribution_method': 'Unknown', 'cp_requirements_met': False,
            'cpk_requirements_met': False, 'requirements_met': False, 'cp_target': cp_target,
            'cpk_target': cpk_target
        }

def create_scatter_plot(param_name: str, measurements: List[float], stats: Dict[str, float], usl: float, lsl: float, dpi: int = 300) -> Optional[str]:
    try:
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
        if dpi < 72 or dpi > 1200:
            raise ValueError("DPI must be between 72 and 1200")

        plt.figure(figsize=(15, 6), dpi=dpi)
        plt.grid(True, linestyle='--', alpha=0.7)

        x = np.arange(1, len(measurements) + 1)
        measurements = np.array(measurements)

        ranges = [
            (lsl, stats['mean'] - 3 * stats['std'], 'red'),
            (stats['mean'] - 3 * stats['std'], stats['mean'] + 3 * stats['std'], 'green'),
            (stats['mean'] + 3 * stats['std'], usl, 'blue')
        ]
        out_of_spec_color = 'purple'

        colors = []
        for m in measurements:
            if m < lsl or m > usl:
                colors.append(out_of_spec_color)
            else:
                for lower, upper, color in ranges:
                    if lower <= m <= upper:
                        colors.append(color)
                        break

        for x_val, y_val, color in zip(x, measurements, colors):
            plt.scatter(x_val, y_val, c=color, s=50, zorder=5)

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

        y_range_padding = 0.1 * (usl - lsl)
        y_min = min(lsl, measurements.min()) - y_range_padding
        y_max = max(usl, measurements.max()) + y_range_padding

        plt.title(f'Value Chart for {param_name}', fontsize=14, pad=10)
        plt.xlabel('Value No.', fontsize=12)
        plt.ylabel(param_name, fontsize=12)
        plt.xlim(0.5, len(measurements) + 1.5)
        plt.ylim(y_min, y_max)

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



def analyze_inspection_data_json(json_data: Union[str, dict]) -> tuple[dict, dict]:
    """Analyze JSON data and return metadata and parameter information with statistical analysis."""
    try:
        # Load JSON data if provided as a string
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        # Extract metadata and parameter_info
        metadata = data.get('metadata', {})
        parameter_info_input = data.get('parameter_info', {})

        parameter_info = {}
        for param_name, param_data in parameter_info_input.items():
            try:
                # Extract required fields
                measurements = param_data.get('measurements', [])
                usl = float(param_data.get('USL', np.nan))
                lsl = float(param_data.get('LSL', np.nan))
                nominal = float(param_data.get('Nominal_Value', np.nan))
                cp_target = float(param_data.get('cp_target', 1.33))
                cpk_target = float(param_data.get('cpk_target', 1.33))
                tolerance = usl - lsl if pd.notna(usl) and pd.notna(lsl) else 'Unknown'

                # Validate measurements
                if not measurements or not all(isinstance(m, (int, float)) for m in measurements):
                    print(f"Invalid or empty measurements for parameter '{param_name}'")
                    continue

                # Create parameter data dictionary
                param_data_output = {
                    'char_number': param_data.get('char_number', 'Unknown'),
                    'char_descr': param_data.get('char_descr', param_name),
                    'char_class': param_data.get('char_class', 'Unknown'),
                    'unit': param_data.get('unit', 'Unknown'),
                    'USL': usl,
                    'LSL': lsl,
                    'Nominal_Value': nominal,
                    'evaluation_starttime': param_data.get('evaluation_starttime', None),
                    'evaluation_endtime': param_data.get('evaluation_endtime', None),
                    'Subgroup_Size': param_data.get('Subgroup_Size', 'Unknown'),
                    'Subgroup_Type': param_data.get('Subgroup_Type', 'Unknown'),
                    'value_count': len(measurements),
                    'Tolerance': tolerance,
                    'measurements': measurements
                }

                # Calculate statistics
                stats = calculate_statistics(measurements, usl, lsl, cp_target, cpk_target)
                param_data_output.update(stats)

                # Generate scatter plot
                param_data_output['plot_image'] = create_scatter_plot(param_name, measurements, stats, usl, lsl)

                parameter_info[param_name] = param_data_output

            except Exception as e:
                print(f"Error processing parameter '{param_name}': {e}")
                continue

        return metadata, parameter_info

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return {}, {}
    except Exception as e:
        print(f"Error analyzing JSON data: {e}")
        return {}, {}
    
def calculate_graph_statistics(measurements: List[float]) -> Dict[str, float]:
    """
    Calculate statistical metrics for a list of measurements.
    Args:
        measurements (List[float]): List of numeric measurements.
    Returns:
        Dict[str, float]: Dictionary with mean, std, max, min, UCL, and LCL.
    Raises:
        ValueError: If measurements are empty or contain non-numeric values.
    """
    if not measurements:
        raise ValueError("Measurements list cannot be empty")
    
    # Check if all measurements are numeric
    if not all(isinstance(x, (int, float)) for x in measurements):
        raise ValueError("All measurements must be numeric")
    
    mean = np.mean(measurements)
    std = np.std(measurements, ddof=1)  # Sample standard deviation
    max_val = np.max(measurements)
    min_val = np.min(measurements)
    n = len(measurements)
    # Calculate UCL and LCL for control charts
    ucl = mean + 3 * (std / sqrt(n))
    lcl = mean - 3 * (std / sqrt(n))
    
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "max": round(max_val, 4),
        "min": round(min_val, 4),
        "ucl": round(ucl, 4),
        "lcl": round(lcl, 4)
    }

# Example Usage
# if __name__ == "__main__":
#     try:
#         metadata, parameter_info = analyze_inspection_data('DI REPORT OF DISC SEPERATOR.xlsx')
#         print("Metadata:", metadata)
#         for param, info in parameter_info.items():
#             print(f"\nParameter: {param}")
#             for key, value in info.items():
#                 print(f"{key}: {value}")
#     except Exception as e:

#         print(f"Error in main execution: {e}")