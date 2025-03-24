#!/usr/bin/env python3
import re
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

class ErrorAnalyzer:
    def __init__(self, ranks_to_analyze=None, max_errors_per_type=3):
        # Ranks to analyze, default is None (all ranks)
        self.ranks_to_analyze = ranks_to_analyze
        # Maximum number of errors to report per type
        self.max_errors_per_type = max_errors_per_type
        
        # Error patterns with their classifications
        self.error_patterns = {
            'CUDA_OOM': {
                'pattern': r'RuntimeError: CUDA out of memory.*?(?=\n\n|\Z)',
                'description': 'CUDA Out of Memory Error',
                'suggestion': 'Consider reducing batch size, model size, or checking for memory leaks'
            },
            'NODE_FAILURE': {
                'patterns': [
                    r'Connection refused.*?|Socket timed out.*?|Node failure detected.*?(?=\n\n|\Z)',
                    r'Node failure on \w+',
                    r'CANCELLED AT .*? DUE TO NODE FAILURE'
                ],
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

    def split_by_rank(self, content: str) -> Dict[int, str]:
        """
        Split error file content by MPI rank to analyze specific ranks.
        Returns a dictionary mapping rank number to content chunks.
        Only includes default0 logs for the specified global ranks.
        """
        # Dictionary to store content by global rank
        rank_contents = {}
        
        # Pattern to identify global rank markers with local rank 0
        local0_global_pattern = r'(?:^|\n)\[default0\](?:\[(?:Rank|RANK|rank)[:\s]+(\d+)\]|\[(\d+)\])'
        
        # Find all matches for default0 with a global rank
        local0_global_matches = list(re.finditer(local0_global_pattern, content, re.MULTILINE))
        
        if local0_global_matches:
            # Process each global rank section (with local rank 0)
            for i, marker in enumerate(local0_global_matches):
                # Extract the global rank - might be in group 1 or 2 depending on format
                global_rank = int(marker.group(1) if marker.group(1) is not None else marker.group(2))
                
                # If we're only interested in specific ranks, skip others
                if self.ranks_to_analyze is not None and global_rank not in self.ranks_to_analyze:
                    continue
                    
                start_pos = marker.start()
                
                # End position is either the start of next match or end of file
                if i < len(local0_global_matches) - 1:
                    end_pos = local0_global_matches[i+1].start()
                else:
                    end_pos = len(content)
                    
                # Add this content to the rank
                if global_rank in rank_contents:
                    rank_contents[global_rank] += content[start_pos:end_pos]
                else:
                    rank_contents[global_rank] = content[start_pos:end_pos]
        
        # If no valid default0+global rank sections were found, fall back to global rank matching
        if not rank_contents:
            # Pattern to match global ranks
            global_rank_pattern = r'(?:^|\n)(?:Rank|RANK|rank)[:\s]+(\d+)'
            global_rank_matches = list(re.finditer(global_rank_pattern, content, re.MULTILINE))
            
            if global_rank_matches:
                for i, marker in enumerate(global_rank_matches):
                    global_rank = int(marker.group(1))
                    
                    # Filter by requested ranks
                    if self.ranks_to_analyze is not None and global_rank not in self.ranks_to_analyze:
                        continue
                        
                    start_pos = marker.start()
                    
                    if i < len(global_rank_matches) - 1:
                        end_pos = global_rank_matches[i+1].start()
                    else:
                        end_pos = len(content)
                        
                    if global_rank in rank_contents:
                        rank_contents[global_rank] += content[start_pos:end_pos]
                    else:
                        rank_contents[global_rank] = content[start_pos:end_pos]
        
        # If still no valid ranks were found but ranks were specified, look for rank-agnostic global errors
        if not rank_contents and self.ranks_to_analyze:
            # Add global errors that don't have rank specifiers
            rank_contents[-1] = content
        
        return rank_contents


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
            
            # Add first 20 lines with numbers
            for i, line in enumerate(first_part):
                numbered_lines.append(f"[L{start_line + i}] {line}")
            
            # Add separator with line count info
            omitted_lines = total_lines - 40
            numbered_lines.append(f"\n... [{omitted_lines} lines omitted] ...\n")
            
            # Add last 20 lines with numbers
            for i, line in enumerate(last_part):
                line_num = start_line + total_lines - 20 + i
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

        # Split content by rank
        rank_contents = self.split_by_rank(content)
        
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error_file': error_file_path,
            'file_size_mb': round(os.path.getsize(error_file_path) / (1024 * 1024), 2),
            'ranks_analyzed': list(rank_contents.keys()),
            'total_ranks': max(rank_contents.keys()) + 1 if rank_contents and max(rank_contents.keys()) >= 0 else 'unknown',
            'rank_results': {},
            'global_errors': [],
            'error_summary': {},
            'summary': ''
        }

        # Process each rank separately
        for rank, rank_content in rank_contents.items():
            rank_results = {
                'detected_errors': [],
                'traceback': self.extract_traceback(rank_content)
            }
            
            # Process each error pattern
            for error_type, error_info in self.error_patterns.items():
                matches = self.find_pattern_matches(rank_content, error_type, error_info)
                
                # Only keep up to max_errors_per_type
                if matches and len(matches) > self.max_errors_per_type:
                    matches = matches[:self.max_errors_per_type]
                    
                if matches:
                    rank_results['detected_errors'].extend(matches)
                    
                    # Update global count of error types
                    if error_type not in results['error_summary']:
                        results['error_summary'][error_type] = {
                            'count': 0,
                            'description': error_info['description'],
                            'suggestion': error_info['suggestion']
                        }
                    results['error_summary'][error_type]['count'] += len(matches)
            
            # Sort errors by position in file
            rank_results['detected_errors'].sort(key=lambda x: x['position'])
            
            if rank == -1:
                # These are global errors not associated with a specific rank
                results['global_errors'] = rank_results['detected_errors']
            else:
                results['rank_results'][rank] = rank_results

        # Create summary
        if results['error_summary'] or results['global_errors']:
            summary_lines = ['Error Analysis Summary:']
            for error_type, info in results['error_summary'].items():
                summary_lines.append(f"- {info['description']}: {info['count']} occurrence(s)")
            if len(results['rank_results']) > 0:
                summary_lines.append(f"\nAnalyzed {len(results['rank_results'])} ranks out of {results['total_ranks']} total ranks")
            results['summary'] = '\n'.join(summary_lines)
        else:
            results['summary'] = 'No known error patterns detected in the log file.'

        return results

    def save_analysis(self, results: Dict, output_file: str):
        """Save the analysis results to a file in a readable format."""
        with open(output_file, 'w') as f:
            f.write(f"Error Analysis Report\n")
            f.write(f"Generated: {results['timestamp']}\n")
            f.write(f"Analyzed File: {results['error_file']} ({results['file_size_mb']} MB)\n")
            
            if 'ranks_analyzed' in results and results['ranks_analyzed']:
                ranks_str = 'all' if results['ranks_analyzed'] == [-1] else ', '.join(map(str, sorted(results['ranks_analyzed'])))
                f.write(f"Ranks Analyzed: {ranks_str}\n")
                if isinstance(results['total_ranks'], int) and results['total_ranks'] > 0:
                    f.write(f"Total Ranks: {results['total_ranks']}\n")
            
            f.write("\n" + "="*50 + "\n\n")
            
            f.write(results['summary'])
            f.write("\n\n" + "="*50 + "\n\n")
            
            # Global errors (not rank-specific)
            if results['global_errors']:
                f.write("Global Errors (not associated with specific ranks):\n\n")
                for i, error in enumerate(results['global_errors'], 1):
                    f.write(f"Error {i}:\n")
                    f.write(f"Type: {error['description']}\n")
                    f.write(f"Matched Pattern: {error['matched_text'][:100]}...(truncated)\n")
                    f.write(f"Suggestion: {error['suggestion']}\n")
                    f.write("Context:\n")
                    f.write("-"*20 + "\n")
                    f.write(error['context'])
                    f.write("\n" + "-"*20 + "\n\n")
                f.write("="*50 + "\n\n")
            
            # Process rank-specific results
            if results['rank_results']:
                for rank, rank_data in sorted(results['rank_results'].items()):
                    f.write(f"RANK {rank} ANALYSIS:\n")
                    f.write("-"*50 + "\n\n")
                    
                    # Write traceback for this rank
                    if rank_data['traceback']['text']:
                        f.write(f"Python Traceback (Rank {rank}):\n")
                        if rank_data['traceback']['full_length'] > 40:
                            f.write(f"[Full traceback length: {rank_data['traceback']['full_length']} lines, "
                                   f"starting at line {rank_data['traceback']['start_line']} in original file]\n")
                        f.write("-"*20 + "\n")
                        f.write(rank_data['traceback']['text'])
                        f.write("\n" + "-"*20 + "\n\n")
                    
                    # Write errors for this rank
                    if rank_data['detected_errors']:
                        f.write(f"Errors for Rank {rank}:\n\n")
                        for i, error in enumerate(rank_data['detected_errors'], 1):
                            f.write(f"Error {i}:\n")
                            f.write(f"Type: {error['description']}\n")
                            # Truncate very long matched patterns
                            f.write(f"Matched Pattern: {error['matched_text'][:100]}..." if len(error['matched_text']) > 100 
                                    else f"Matched Pattern: {error['matched_text']}\n")
                            f.write(f"Suggestion: {error['suggestion']}\n")
                            f.write("Context:\n")
                            f.write("-"*20 + "\n")
                            f.write(error['context'])
                            f.write("\n" + "-"*20 + "\n\n")
                    else:
                        f.write(f"No detailed errors to report for Rank {rank}.\n")
                    
                    f.write("\n" + "="*50 + "\n\n")
            
            # Add recommendations based on error types
            if results['error_summary']:
                f.write("RECOMMENDATIONS:\n")
                for error_type, info in results['error_summary'].items():
                    if info['count'] > 0:
                        f.write(f"For {info['description']} ({info['count']} occurrences):\n")
                        f.write(f"- {info['suggestion']}\n\n")


def get_job_id(filename):
    # Try to extract job ID from slurm format first (%x-%j.err)
    match = re.search(r'-(\d{6,})\.err', filename)
    if not match:
        # Fallback: look for any 6+ digit number in filename
        match = re.search(r'(\d{6,})', filename)
    return match.group(1) if match else 'unknown'


def main():
    parser = argparse.ArgumentParser(description='Analyze error files from distributed training jobs')
    parser.add_argument('error_file', help='Path to the error file to analyze')
    parser.add_argument('--ranks', type=str, default='0,1', 
                        help='Comma-separated list of ranks to analyze (default: 0,1)')
    parser.add_argument('--max-errors', type=int, default=3,
                        help='Maximum number of errors to report per type (default: 3)')
    parser.add_argument('--all-ranks', action='store_true',
                        help='Analyze all ranks (overrides --ranks)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: auto-generated based on input filename)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.error_file):
        print(f"Error file not found: {args.error_file}")
        sys.exit(1)
    
    # Parse ranks to analyze
    ranks_to_analyze = None if args.all_ranks else [int(r) for r in args.ranks.split(',') if r.strip()]
    
    analyzer = ErrorAnalyzer(ranks_to_analyze=ranks_to_analyze, max_errors_per_type=args.max_errors)
    results = analyzer.analyze_error_file(args.error_file)
    
    # Create output file name
    if args.output:
        output_file = args.output
    else:
        job_id = get_job_id(args.error_file)
        output_file = f"failure-report-{job_id}.txt"
    
    analyzer.save_analysis(results, output_file)
    print(f"Analysis saved to: {output_file}")
    
    # Print brief summary to console
    print("\nBrief Summary:")
    print("-" * 40)
    for line in results['summary'].split('\n'):
        print(line)


if __name__ == "__main__":
    main()