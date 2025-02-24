#!/usr/bin/env python3
import re
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class ErrorAnalyzer:
    def __init__(self):
        # Error patterns with their classifications
        self.error_patterns = {
            'CUDA_OOM': {
                'pattern': r'RuntimeError: CUDA out of memory.*?(?=\n\n|\Z)',
                'description': 'CUDA Out of Memory Error',
                'suggestion': 'Consider reducing batch size, model size, or checking for memory leaks'
            },
            'NODE_FAILURE': {
                'pattern': r'Connection refused.*?|Socket timed out.*?|Node failure detected.*?(?=\n\n|\Z)',
                'description': 'Node Communication/Failure Error',
                'suggestion': 'Check cluster health and network connectivity'
            },
            'COLLECTIVE_OPERATION': {
                'primary_pattern': r'ProcessGroupNCCL.*?(?=\n)',
                'secondary_patterns': [
                    r'Timeout\(ms\)=\d+',
                    r'\d+ milliseconds before timing out',
                    r'Some NCCL operations have failed or timed out.*?corrupted/incomplete data',
                    r'[Aa]ll[Rr]educe.*?(?=\n)',
                    r'[Aa]ll[Gg]ather.*?(?=\n)',
                    r'[Bb]roadcast.*?(?=\n)',
                    r'[Rr]educe[Ss]catter.*?(?=\n)'
                ],
                'description': 'Collective Operation Failure',
                'suggestion': 'Check network connectivity, increase timeout values, or investigate node synchronization issues'
            },
            'TIME_LIMIT': {
                'patterns': [
                    r'slurmstepd: error: \*\*\* JOB \d+ ON .+ CANCELLED AT .+ DUE TO TIME LIMIT \*\*\*',
                    r'CANCELLED AT .*? DUE TO TIME LIMIT'
                ],
                'description': 'Job Time Limit Exceeded',
                'suggestion': 'Increase job time limit or optimize training'
            },
            'JOB_CANCELLED': {
                'patterns': [
                    r'srun: Job step aborted',
                    r'srun: forcing job termination',
                    r'CANCELLED AT .*? BY \w+'
                ],
                'description': 'Job Cancelled by User/Admin',
                'suggestion': 'Job was manually terminated'
            },
            'PYTHON_TYPE_ERROR': {
                'pattern': r'TypeError:.*?(?=\n\n|\Z)',
                'description': 'Python Type Error',
                'suggestion': 'Check data types and function arguments'
            },
            'PYTHON_VALUE_ERROR': {
                'pattern': r'ValueError:.*?(?=\n\n|\Z)',
                'description': 'Python Value Error',
                'suggestion': 'Verify input values and ranges'
            },
            'PYTHON_ATTRIBUTE_ERROR': {
                'pattern': r'AttributeError:.*?(?=\n\n|\Z)',
                'description': 'Python Attribute Error',
                'suggestion': 'Check object attributes and method calls'
            },
            'PYTHON_INDEX_ERROR': {
                'pattern': r'IndexError:.*?(?=\n\n|\Z)',
                'description': 'Python Index Error',
                'suggestion': 'Verify array/list indices and bounds'
            },
            'PYTHON_KEY_ERROR': {
                'pattern': r'KeyError:.*?(?=\n\n|\Z)',
                'description': 'Python Key Error',
                'suggestion': 'Check dictionary keys and access'
            },
            'PYTHON_NAME_ERROR': {
                'pattern': r'NameError:.*?(?=\n\n|\Z)',
                'description': 'Python Name Error',
                'suggestion': 'Verify variable names and scope'
            },
            'PYTHON_ZERO_DIVISION': {
                'pattern': r'ZeroDivisionError:.*?(?=\n\n|\Z)',
                'description': 'Python Zero Division Error',
                'suggestion': 'Check for division by zero conditions'
            },
            'PYTHON_ASSERTION': {
                'pattern': r'AssertionError:.*?(?=\n\n|\Z)',
                'description': 'Python Assertion Error',
                'suggestion': 'Check assertion conditions and expected values'
            },
            'PYTHON_OVERFLOW': {
                'pattern': r'OverflowError:.*?(?=\n\n|\Z)',
                'description': 'Python Overflow Error',
                'suggestion': 'Check numerical operations and value ranges'
            },
            'PYTHON_RUNTIME': {
                'pattern': r'RuntimeError:(?!.*CUDA out of memory).*?(?=\n\n|\Z)',  # Exclude CUDA OOM which is handled separately
                'description': 'Python Runtime Error',
                'suggestion': 'Check program logic and runtime conditions'
            },
            'PYTHON_IMPORT': {
                'patterns': [
                    r'ImportError:.*?(?=\n\n|\Z)',
                    r'ModuleNotFoundError:.*?(?=\n\n|\Z)'
                ],
                'description': 'Python Import/Module Error',
                'suggestion': 'Check package installation and Python environment'
            },
            'PERMISSION_ERROR': {
                'pattern': r'PermissionError:.*?(?=\n\n|\Z)',
                'description': 'Permission Error',
                'suggestion': 'Check file/directory permissions and user access rights'
            },
            'FILE_ERROR': {
                'patterns': [
                    r'FileNotFoundError:.*?(?=\n\n|\Z)',
                    r'IOError:.*?(?=\n\n|\Z)',
                    r'OSError:.*?(?=\n\n|\Z)'
                ],
                'description': 'File System Error',
                'suggestion': 'Check file paths, permissions, and disk space'
            },
            'GPU_ERROR': {
                'pattern': r'CUDA driver version.*?|CUDA error:.*?(?=\n\n|\Z)',
                'description': 'GPU/CUDA Error',
                'suggestion': 'Verify CUDA installation and GPU health'
            },
            'MEMORY_ERROR': {
                'pattern': r'MemoryError:.*?(?=\n\n|\Z)',
                'description': 'Python Memory Error',
                'suggestion': 'Check memory usage and available system memory'
            }
        }

    def extract_traceback(self, content: str) -> Dict[str, str]:
        """
        Extract Python traceback if present, with line numbers.
        For long tracebacks, show first and last 10 lines with line numbers.
        """
        traceback_pattern = r'Traceback \(most recent call last\):.*?(?=\n\n|\Z)'
        match = re.search(traceback_pattern, content, re.DOTALL)
        
        if not match:
            return {'text': '', 'start_line': 0, 'full_length': 0}
            
        traceback = match.group(0)
        
        # Find the starting line number in the original file
        start_pos = match.start()
        start_line = content.count('\n', 0, start_pos) + 1
        
        # Split traceback into lines
        lines = traceback.split('\n')
        total_lines = len(lines)
        
        if total_lines > 40:
            # Keep first 20 and last 20 lines
            first_part = lines[:20]
            last_part = lines[-20:]
            
            # Create the condensed traceback with line numbers
            numbered_lines = []
            
            # Add first 10 lines with numbers
            for i, line in enumerate(first_part):
                numbered_lines.append(f"[L{start_line + i}] {line}")
            
            # Add separator with line count info
            omitted_lines = total_lines - 40
            numbered_lines.append(f"\n... [{omitted_lines} lines omitted] ...\n")
            
            # Add last 10 lines with numbers
            for i, line in enumerate(last_part):
                line_num = start_line + total_lines - 15 + i
                numbered_lines.append(f"[L{line_num}] {line}")
            
            formatted_traceback = '\n'.join(numbered_lines)
        else:
            # Number all lines if traceback is short
            numbered_lines = [f"[L{start_line + i}] {line}" 
                            for i, line in enumerate(lines)]
            formatted_traceback = '\n'.join(numbered_lines)
        
        return {
            'text': formatted_traceback,
            'start_line': start_line,
            'full_length': total_lines
        }

    def extract_error_context(self, content: str, error_pos: int, context_lines: int = 5) -> str:
        """Extract lines before and after the error position for context."""
        lines = content.split('\n')
        cumulative_length = 0
        error_line_num = 0
        
        # Find the error line number
        for i, line in enumerate(lines):
            cumulative_length += len(line) + 1  # +1 for newline
            if cumulative_length > error_pos:
                error_line_num = i
                break
        
        start = max(0, error_line_num - context_lines)
        end = min(len(lines), error_line_num + context_lines + 1)
        
        return '\n'.join(lines[start:end])

    def find_pattern_matches(self, content: str, error_type: str, error_info: Dict) -> List[Dict]:
        """Find matches for patterns, with special handling for collective operations."""
        matches = []
        
        # Special handling for collective operations requiring multiple patterns
        if error_type == 'COLLECTIVE_OPERATION':
            # First find ProcessGroupNCCL matches
            primary_matches = list(re.finditer(error_info['primary_pattern'], content, re.MULTILINE | re.DOTALL))
            
            if primary_matches:
                # For each ProcessGroupNCCL match, look for at least one secondary pattern nearby
                for primary_match in primary_matches:
                    start_pos = max(0, primary_match.start() - 500)  # Look up to 500 chars before
                    end_pos = min(len(content), primary_match.end() + 500)  # Look up to 500 chars after
                    search_region = content[start_pos:end_pos]
                    
                    # Check for any secondary pattern
                    for secondary_pattern in error_info['secondary_patterns']:
                        if re.search(secondary_pattern, search_region, re.MULTILINE | re.DOTALL):
                            # Found a valid combination of primary and secondary patterns
                            matches.append({
                                'type': error_type,
                                'description': error_info['description'],
                                'suggestion': error_info['suggestion'],
                                'context': self.extract_error_context(content, primary_match.start()),
                                'position': primary_match.start(),
                                'matched_text': f"ProcessGroupNCCL with {re.search(secondary_pattern, search_region).group(0)}"
                            })
                            break  # 1 secondary pattern is enough
        
        # Standard pattern matching for other error types
        else:
            patterns = error_info.get('patterns', [error_info.get('pattern')])
            if not isinstance(patterns, list):
                patterns = [patterns]
                
            for pattern in patterns:
                for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
                    matches.append({
                        'type': error_type,
                        'description': error_info['description'],
                        'suggestion': error_info['suggestion'],
                        'context': self.extract_error_context(content, match.start()),
                        'position': match.start(),
                        'matched_text': match.group(0)
                    })
                    
        return matches


    def analyze_error_file(self, error_file_path: str) -> Dict:
        """Analyze the error file and return classified errors with context."""
        try:
            with open(error_file_path, 'r') as f:
                content = f.read()
        except Exception as e:
            return {'error': f'Failed to read error file: {str(e)}'}

        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error_file': error_file_path,
            'detected_errors': [],
            'traceback': self.extract_traceback(content),
            'summary': ''
        }

        found_any = False
        for error_type, error_info in self.error_patterns.items():
            matches = self.find_pattern_matches(content, error_type, error_info)
            if matches:
                found_any = True
                results['detected_errors'].extend(matches)

        # Sort errors by position in file
        results['detected_errors'].sort(key=lambda x: x['position'])

        # Create summary
        if found_any:
            error_counts = {}
            for error in results['detected_errors']:
                error_counts[error['type']] = error_counts.get(error['type'], 0) + 1
            
            summary_lines = ['Error Analysis Summary:']
            summary_lines.extend([f"- {self.error_patterns[k]['description']}: {v} occurrence(s)"
                                for k, v in error_counts.items()])
            results['summary'] = '\n'.join(summary_lines)
        else:
            results['summary'] = 'No known error patterns detected in the log file.'

        return results

    def save_analysis(self, results: Dict, output_file: str):
        """Save the analysis results to a file in a readable format."""
        with open(output_file, 'w') as f:
            f.write(f"Error Analysis Report\n")
            f.write(f"Generated: {results['timestamp']}\n")
            f.write(f"Analyzed File: {results['error_file']}\n")
            f.write("\n" + "="*50 + "\n\n")
            
            f.write(results['summary'])
            f.write("\n\n" + "="*50 + "\n\n")
            
            if results['traceback']['text']:
                f.write("Python Traceback:\n")
                if results['traceback']['full_length'] > 40:
                    f.write(f"[Full traceback length: {results['traceback']['full_length']} lines, "
                           f"starting at line {results['traceback']['start_line']} in original file]\n")
                f.write("-"*20 + "\n")
                f.write(results['traceback']['text'])
                f.write("\n" + "-"*20 + "\n\n")
            
            if results['detected_errors']:
                f.write("Detailed Error Analysis:\n\n")
                for i, error in enumerate(results['detected_errors'], 1):
                    f.write(f"Error {i}:\n")
                    f.write(f"Type: {error['description']}\n")
                    f.write(f"Matched Pattern: {error['matched_text']}\n")
                    f.write(f"Suggestion: {error['suggestion']}\n")
                    f.write("Context:\n")
                    f.write("-"*20 + "\n")
                    f.write(error['context'])
                    f.write("\n" + "_"*100 + "\n\n\n")
            else:
                f.write("No detailed errors to report.\n")


def get_job_id(filename):
    # Try to extract job ID from slurm format first (%x-%j.err)
    match = re.search(r'-(\d{6,})\.err', filename)
    if not match:
        # Fallback: look for any 6+ digit number in filename
        match = re.search(r'(\d{6,})', filename)
    return match.group(1) if match else 'unknown'


def main():
    if len(sys.argv) != 2:
        print("Usage: python error_analyzer.py <error_file_path>")
        sys.exit(1)

    error_file = sys.argv[1]
    if not os.path.exists(error_file):
        print(f"Error file not found: {error_file}")
        sys.exit(1)

    analyzer = ErrorAnalyzer()
    results = analyzer.analyze_error_file(error_file)
    
    # Create output file name based on input file
    job_id = get_job_id(error_file)
    output_file = f"failure-report-{job_id}.err"
    analyzer.save_analysis(results, output_file)
    print(f"Analysis saved to: {output_file}")

if __name__ == "__main__":
    main()