"""
Accuracy Analysis Script for Patient2TrialFlow and Trial2PatientFlow Testing
Analyzes consistency and accuracy across 20 runs
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter


class TestingAccuracyAnalyzer:
    def __init__(self, testing_dir: Path):
        """
        Initialize analyzer
        
        Args:
            testing_dir: Path to Patient2TrialFlowTesting or Trial2PatientFlowTesting directory
        """
        self.testing_dir = Path(testing_dir)
        if not self.testing_dir.exists():
            raise ValueError(f"Testing directory not found: {testing_dir}")
        
        # Determine flow type from directory name
        self.flow_type = "Patient2Trial" if "Patient2Trial" in str(testing_dir) else "Trial2Patient"
        
        # Load all run results
        self.runs_data = []
        self.load_all_runs()
    
    def load_all_runs(self):
        """Load results from all run directories"""
        print(f"Loading results from {self.testing_dir}...")
        
        for run_num in range(1, 21):  # Runs 1-20
            run_dir = self.testing_dir / f"Run{run_num}"
            result_dir = run_dir / "result"
            
            if not result_dir.exists():
                print(f"[WARNING] Run {run_num} result directory not found")
                continue
            
            # Find result JSON file
            result_files = list(result_dir.glob("run*_results.json"))
            if not result_files:
                print(f"[WARNING] No result file found in Run {run_num}")
                continue
            
            result_file = result_files[0]
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    self.runs_data.append({
                        'run_num': run_num,
                        'results': results
                    })
                print(f"[OK] Loaded Run {run_num}")
            except Exception as e:
                print(f"[ERROR] Error loading Run {run_num}: {e}")
        
        print(f"\n[OK] Loaded {len(self.runs_data)} runs successfully")
    
    def extract_evaluations_from_run(self, run_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract evaluation data from a run's results"""
        evaluations = []
        results = run_data['results']
        
        # Try different paths to find final_ranking
        final_ranking = None
        if 'steps' in results:
            if 'patient_trial_matching' in results['steps']:
                final_ranking = results['steps']['patient_trial_matching'].get('results', {}).get('final_ranking', [])
            elif 'trial_patient_matching' in results['steps']:
                final_ranking = results['steps']['trial_patient_matching'].get('results', {}).get('final_ranking', [])
        
        if not final_ranking:
            final_ranking = results.get('final_ranking', [])
        
        for eval_item in final_ranking:
            if self.flow_type == "Patient2Trial":
                trial_info = eval_item.get('trial_info', {})
                patient_info = eval_item.get('patient_info', {})
                evaluations.append({
                    'patient_id': patient_info.get('patient_id', 'Unknown'),
                    'trial_id': trial_info.get('trial_id', 'Unknown'),
                    'eligibility': eval_item.get('eligibility_status', 'UNKNOWN'),
                    'confidence': eval_item.get('confidence_score', 0),
                    'reasoning': eval_item.get('reasoning', 'N/A')
                })
            else:  # Trial2Patient
                patient_info = eval_item.get('patient_info', {})
                trial_info = eval_item.get('trial_info', {})
                evaluations.append({
                    'trial_id': trial_info.get('trial_id', 'Unknown'),
                    'patient_id': patient_info.get('patient_id', 'Unknown'),
                    'eligibility': eval_item.get('eligibility_status', 'UNKNOWN'),
                    'confidence': eval_item.get('confidence_score', 0),
                    'reasoning': eval_item.get('reasoning', 'N/A')
                })
        
        return evaluations
    
    def calculate_accuracy_metrics(self) -> Dict[str, Any]:
        """Calculate accuracy and consistency metrics"""
        print("\n" + "="*80)
        print("CALCULATING ACCURACY METRICS")
        print("="*80)
        
        # Group evaluations by entity (trial_id for Patient2Trial, patient_id for Trial2Patient)
        entity_evaluations = defaultdict(list)  # entity_id -> list of evaluations across runs
        
        for run_data in self.runs_data:
            evaluations = self.extract_evaluations_from_run(run_data)
            run_num = run_data['run_num']
            
            for eval_item in evaluations:
                if self.flow_type == "Patient2Trial":
                    entity_id = eval_item['trial_id']
                else:
                    entity_id = str(eval_item['patient_id'])
                
                entity_evaluations[entity_id].append({
                    'run_num': run_num,
                    'eligibility': eval_item['eligibility'],
                    'confidence': eval_item['confidence']
                })
        
        # Calculate metrics for each entity
        entity_metrics = {}
        overall_stats = {
            'total_entities': len(entity_evaluations),
            'total_evaluations': sum(len(evals) for evals in entity_evaluations.values()),
            'consistent_entities': 0,  # Entities with same status in all runs
            'highly_consistent_entities': 0,  # Entities with >= 90% consistency
            'moderately_consistent_entities': 0,  # Entities with 70-90% consistency
            'low_consistent_entities': 0,  # Entities with < 70% consistency
            'eligibility_distribution': Counter(),
            'average_consistency': 0,
            'average_confidence': 0
        }
        
        all_consistencies = []
        all_confidences = []
        
        for entity_id, evaluations in entity_evaluations.items():
            # Count status occurrences
            statuses = [e['eligibility'] for e in evaluations]
            status_counter = Counter(statuses)
            
            # Most common status
            most_common_status, most_common_count = status_counter.most_common(1)[0]
            
            # Consistency score (percentage of runs with most common status)
            consistency_score = (most_common_count / len(evaluations)) * 100 if evaluations else 0
            
            # Average confidence
            confidences = [e['confidence'] for e in evaluations if e['confidence']]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Status changes count
            status_changes = 0
            prev_status = None
            for status in statuses:
                if prev_status is not None and status != prev_status:
                    status_changes += 1
                prev_status = status
            
            entity_metrics[entity_id] = {
                'total_runs': len(evaluations),
                'eligibility_distribution': dict(status_counter),
                'most_common_status': most_common_status,
                'most_common_count': most_common_count,
                'consistency_score': consistency_score,
                'status_changes': status_changes,
                'average_confidence': avg_confidence,
                'is_fully_consistent': consistency_score == 100.0
            }
            
            # Update overall stats
            all_consistencies.append(consistency_score)
            all_confidences.extend(confidences)
            overall_stats['eligibility_distribution'].update(statuses)
            
            if consistency_score == 100.0:
                overall_stats['consistent_entities'] += 1
            elif consistency_score >= 90:
                overall_stats['highly_consistent_entities'] += 1
            elif consistency_score >= 70:
                overall_stats['moderately_consistent_entities'] += 1
            else:
                overall_stats['low_consistent_entities'] += 1
        
        # Calculate overall averages
        overall_stats['average_consistency'] = sum(all_consistencies) / len(all_consistencies) if all_consistencies else 0
        overall_stats['average_confidence'] = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        overall_stats['eligibility_distribution'] = dict(overall_stats['eligibility_distribution'])
        
        return {
            'entity_metrics': entity_metrics,
            'overall_stats': overall_stats,
            'flow_type': self.flow_type
        }
    
    def generate_accuracy_report(self, metrics: Dict[str, Any]) -> Path:
        """Generate Excel report with accuracy analysis"""
        print("\n" + "="*80)
        print("GENERATING ACCURACY REPORT")
        print("="*80)
        
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.testing_dir / f"Accuracy_Report_{self.flow_type}_{timestamp}.xlsx"
        wb = pd.ExcelWriter(report_path, engine='openpyxl')
        
        # Sheet 1: Overall Summary
        overall_stats = metrics['overall_stats']
        summary_data = {
            'Metric': [
                'Total Entities Evaluated',
                'Total Evaluations (across all runs)',
                'Fully Consistent Entities (100%)',
                'Highly Consistent Entities (≥90%)',
                'Moderately Consistent Entities (70-90%)',
                'Low Consistent Entities (<70%)',
                'Average Consistency Score',
                'Average Confidence Score',
                'ELIGIBLE Count',
                'NOT_ELIGIBLE Count',
                'NEED_MORE_INFO Count'
            ],
            'Value': [
                overall_stats['total_entities'],
                overall_stats['total_evaluations'],
                overall_stats['consistent_entities'],
                overall_stats['highly_consistent_entities'],
                overall_stats['moderately_consistent_entities'],
                overall_stats['low_consistent_entities'],
                f"{overall_stats['average_consistency']:.2f}%",
                f"{overall_stats['average_confidence']:.2f}%",
                overall_stats['eligibility_distribution'].get('ELIGIBLE', 0),
                overall_stats['eligibility_distribution'].get('NOT_ELIGIBLE', 0),
                overall_stats['eligibility_distribution'].get('NEED_MORE_INFO', 0)
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(wb, sheet_name='Overall Summary', index=False)
        
        # Sheet 2: Entity-level Metrics
        entity_metrics = metrics['entity_metrics']
        entity_data = []
        for entity_id, metric in entity_metrics.items():
            entity_data.append({
                'Entity ID': entity_id,
                'Total Runs': metric['total_runs'],
                'Most Common Status': metric['most_common_status'],
                'Most Common Count': metric['most_common_count'],
                'Consistency Score (%)': f"{metric['consistency_score']:.2f}",
                'Status Changes': metric['status_changes'],
                'Average Confidence': f"{metric['average_confidence']:.2f}",
                'ELIGIBLE Count': metric['eligibility_distribution'].get('ELIGIBLE', 0),
                'NOT_ELIGIBLE Count': metric['eligibility_distribution'].get('NOT_ELIGIBLE', 0),
                'NEED_MORE_INFO Count': metric['eligibility_distribution'].get('NEED_MORE_INFO', 0),
                'Fully Consistent': 'Yes' if metric['is_fully_consistent'] else 'No'
            })
        
        df_entities = pd.DataFrame(entity_data)
        df_entities = df_entities.sort_values('Consistency Score (%)', ascending=False)
        df_entities.to_excel(wb, sheet_name='Entity Metrics', index=False)
        
        # Sheet 3: Consistency Distribution
        consistency_ranges = {
            '100% (Fully Consistent)': 0,
            '90-99%': 0,
            '80-89%': 0,
            '70-79%': 0,
            '60-69%': 0,
            '50-59%': 0,
            '<50%': 0
        }
        
        for entity_id, metric in entity_metrics.items():
            score = metric['consistency_score']
            if score == 100:
                consistency_ranges['100% (Fully Consistent)'] += 1
            elif score >= 90:
                consistency_ranges['90-99%'] += 1
            elif score >= 80:
                consistency_ranges['80-89%'] += 1
            elif score >= 70:
                consistency_ranges['70-79%'] += 1
            elif score >= 60:
                consistency_ranges['60-69%'] += 1
            elif score >= 50:
                consistency_ranges['50-59%'] += 1
            else:
                consistency_ranges['<50%'] += 1
        
        df_distribution = pd.DataFrame({
            'Consistency Range': list(consistency_ranges.keys()),
            'Number of Entities': list(consistency_ranges.values()),
            'Percentage': [f"{(v/overall_stats['total_entities']*100):.2f}%" for v in consistency_ranges.values()]
        })
        df_distribution.to_excel(wb, sheet_name='Consistency Distribution', index=False)
        
        wb.close()
        
        print(f"[OK] Accuracy report saved to: {report_path}")
        
        return report_path
    
    def print_summary(self, metrics: Dict[str, Any]):
        """Print summary to console"""
        overall_stats = metrics['overall_stats']
        
        print("\n" + "="*80)
        print("ACCURACY ANALYSIS SUMMARY")
        print("="*80)
        print(f"\nFlow Type: {metrics['flow_type']}")
        print(f"\nOverall Statistics:")
        print(f"  Total Entities Evaluated: {overall_stats['total_entities']}")
        print(f"  Total Evaluations (across all runs): {overall_stats['total_evaluations']}")
        print(f"  Average Consistency Score: {overall_stats['average_consistency']:.2f}%")
        print(f"  Average Confidence Score: {overall_stats['average_confidence']:.2f}%")
        print(f"\nConsistency Breakdown:")
        print(f"  Fully Consistent (100%): {overall_stats['consistent_entities']} entities")
        print(f"  Highly Consistent (>=90%): {overall_stats['highly_consistent_entities']} entities")
        print(f"  Moderately Consistent (70-90%): {overall_stats['moderately_consistent_entities']} entities")
        print(f"  Low Consistent (<70%): {overall_stats['low_consistent_entities']} entities")
        print(f"\nEligibility Distribution:")
        for status, count in overall_stats['eligibility_distribution'].items():
            percentage = (count / overall_stats['total_evaluations']) * 100
            print(f"  {status}: {count} ({percentage:.2f}%)")
    
    def analyze(self):
        """Run complete analysis"""
        if not self.runs_data:
            print("[ERROR] No run data found. Cannot perform analysis.")
            return
        
        # Calculate metrics
        metrics = self.calculate_accuracy_metrics()
        
        # Print summary
        self.print_summary(metrics)
        
        # Generate report
        report_path = self.generate_accuracy_report(metrics)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Report saved to: {report_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze accuracy and consistency of testing runs")
    parser.add_argument("--testing-dir", type=str, required=True, 
                       help="Path to Patient2TrialFlowTesting or Trial2PatientFlowTesting directory")
    
    args = parser.parse_args()
    
    try:
        analyzer = TestingAccuracyAnalyzer(Path(args.testing_dir))
        analyzer.analyze()
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

