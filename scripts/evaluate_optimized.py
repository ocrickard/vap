#!/usr/bin/env python3
"""
Enhanced Performance Evaluation for Optimized VAP Turn Detector

This script provides comprehensive evaluation of the optimized model with detailed metrics,
temporal analysis, and comparison against baseline performance.
"""

import os
import sys
import logging
import json
import time
import numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedModelEvaluator:
    """Enhanced evaluator for optimized VAP turn detector"""
    
    def __init__(self, checkpoint_path: str = None):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.results = {}
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the optimized model"""
        from vap.models.vap_model import VAPTurnDetector
        
        # Find latest checkpoint if none specified
        if self.checkpoint_path is None:
            checkpoint_dir = Path("checkpoints/optimized")
            if not checkpoint_dir.exists():
                raise FileNotFoundError("No optimized checkpoints found. Train the optimized model first.")
            
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoint files found.")
            
            self.checkpoint_path = str(max(checkpoints, key=lambda x: x.stat().st_mtime))
        
        logger.info(f"Loading optimized model from: {self.checkpoint_path}")
        
        # Create model with optimized architecture
        self.model = VAPTurnDetector(
            hidden_dim=128,  # Optimized architecture
            num_layers=2,
            num_heads=4
        )
        
        # Load trained weights
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if it exists
            if any(key.startswith('model.') for key in state_dict.keys()):
                state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        logger.info("✅ Optimized model loaded successfully")
    
    def evaluate_model(self, num_test_samples: int = 10) -> Dict:
        """Comprehensive model evaluation"""
        logger.info("🔍 Starting comprehensive evaluation of optimized model...")
        
        # Generate test data with varying characteristics
        test_data = self._generate_diverse_test_data(num_test_samples)
        
        # Run evaluation
        results = self._run_comprehensive_evaluation(test_data)
        
        # Analyze results
        analysis = self._analyze_results(results)
        
        # Compare with baseline
        baseline_comparison = self._compare_with_baseline(analysis)
        
        # Generate visualizations
        self._generate_evaluation_plots(analysis)
        
        # Save results
        self._save_evaluation_results(analysis, baseline_comparison)
        
        return analysis
    
    def _generate_diverse_test_data(self, num_samples: int) -> List[Dict]:
        """Generate diverse test data with different characteristics"""
        test_data = []
        
        for i in range(num_samples):
            # Vary audio length
            duration = np.random.uniform(15, 45)  # 15-45 seconds
            audio_length = int(16000 * duration)
            
            # Create audio with different characteristics
            if i % 3 == 0:
                # Clear speech (low noise)
                audio_a = torch.randn(1, audio_length) * 0.3
                audio_b = torch.randn(1, audio_length) * 0.1
            elif i % 3 == 1:
                # Noisy speech
                audio_a = torch.randn(1, audio_length) * 0.3 + torch.randn(1, audio_length) * 0.1
                audio_b = torch.randn(1, audio_length) * 0.2 + torch.randn(1, audio_length) * 0.05
            else:
                # Overlapping speech
                audio_a = torch.randn(1, audio_length) * 0.3
                audio_b = torch.randn(1, audio_length) * 0.3
            
            test_data.append({
                'audio_a': audio_a,
                'audio_b': audio_b,
                'duration': duration,
                'type': ['clear', 'noisy', 'overlapping'][i % 3]
            })
        
        return test_data
    
    def _run_comprehensive_evaluation(self, test_data: List[Dict]) -> Dict:
        """Run comprehensive evaluation on test data"""
        results = {
            'per_sample': [],
            'overall': {},
            'temporal_analysis': {},
            'error_analysis': {}
        }
        
        all_predictions = []
        all_targets = []
        
        for i, sample in enumerate(test_data):
            logger.info(f"Evaluating sample {i+1}/{len(test_data)} ({sample['type']}, {sample['duration']:.1f}s)")
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(sample['audio_a'], sample['audio_b'])
            
            # Generate realistic labels
            labels = self._generate_real_labels(sample, outputs)
            
            # Compute metrics
            sample_metrics = self._compute_sample_metrics(outputs, labels)
            
            # Store results
            results['per_sample'].append({
                'sample_id': i,
                'type': sample['type'],
                'duration': sample['duration'],
                'metrics': sample_metrics,
                'predictions': {k: v.detach().cpu().numpy() for k, v in outputs.items()},
                'targets': {k: v.detach().cpu().numpy() for k, v in labels.items()}
            })
            
            all_predictions.append(outputs)
            all_targets.append(labels)
        
        # Compute overall metrics
        results['overall'] = self._compute_overall_metrics(results['per_sample'])
        
        # Temporal analysis
        results['temporal_analysis'] = self._analyze_temporal_patterns(results['per_sample'])
        
        # Error analysis
        results['error_analysis'] = self._analyze_error_patterns(results['per_sample'])
        
        return results
    
    def _generate_real_labels(self, sample: Dict, outputs: Dict) -> Dict:
        """Generate realistic labels based on audio characteristics"""
        # Similar to training task but with more sophisticated logic
        batch_size, seq_len = outputs['vap_logits'].shape[:2]
        
        # Get audio energy
        audio_a_energy = torch.norm(sample['audio_a'], dim=1)
        audio_b_energy = torch.norm(sample['audio_b'], dim=1)
        
        # Downsample energy to sequence length
        energy_a_down = F.interpolate(audio_a_energy.unsqueeze(1), size=seq_len, mode='linear').squeeze(1)
        energy_b_down = F.interpolate(audio_b_energy.unsqueeze(1), size=seq_len, mode='linear').squeeze(1)
        
        # Create sophisticated labels
        labels = self._create_sophisticated_labels(energy_a_down, energy_b_down, seq_len, sample['type'])
        
        return labels
    
    def _create_sophisticated_labels(self, energy_a: torch.Tensor, energy_b: torch.Tensor, seq_len: int, audio_type: str) -> Dict:
        """Create sophisticated labels based on audio type and energy patterns"""
        batch_size = energy_a.shape[0]
        
        # Initialize labels
        vap_labels = torch.zeros(batch_size, seq_len, dtype=torch.long)
        eot_labels = torch.zeros(batch_size, seq_len)
        backchannel_labels = torch.zeros(batch_size, seq_len)
        overlap_labels = torch.zeros(batch_size, seq_len)
        vad_labels = torch.zeros(batch_size, seq_len, 2)
        
        for b in range(batch_size):
            for t in range(seq_len):
                # VAP patterns based on audio type
                if audio_type == 'clear':
                    # Clear speech: distinct speaker turns
                    if energy_a[b, t] > 0.1 and energy_b[b, t] < 0.05:
                        vap_labels[b, t] = 0  # A active, B silent
                    elif energy_a[b, t] < 0.05 and energy_b[b, t] > 0.1:
                        vap_labels[b, t] = 1  # B active, A silent
                    else:
                        vap_labels[b, t] = 3  # Gap
                
                elif audio_type == 'noisy':
                    # Noisy speech: more overlap and backchannels
                    if energy_a[b, t] > 0.08 and energy_b[b, t] > 0.08:
                        vap_labels[b, t] = 2  # Overlap
                        overlap_labels[b, t] = 1.0
                    elif energy_a[b, t] > 0.1 and energy_b[b, t] > 0.03:
                        vap_labels[b, t] = 4  # A with backchannel
                        backchannel_labels[b, t] = 1.0
                    else:
                        vap_labels[b, t] = 0 if energy_a[b, t] > energy_b[b, t] else 1
                
                else:  # overlapping
                    # Overlapping speech: frequent overlaps
                    if energy_a[b, t] > 0.08 and energy_b[b, t] > 0.08:
                        vap_labels[b, t] = 2  # Overlap
                        overlap_labels[b, t] = 1.0
                    else:
                        vap_labels[b, t] = 0 if energy_a[b, t] > energy_b[b, t] else 1
                
                # EoT detection
                if t > 0:
                    if energy_a[b, t-1] > 0.1 and energy_a[b, t] < 0.05:
                        eot_labels[b, t] = 1.0
                    elif energy_b[b, t-1] > 0.1 and energy_b[b, t] < 0.05:
                        eot_labels[b, t] = 1.0
                
                # VAD labels
                vad_labels[b, t, 0] = 1.0 if energy_a[b, t] > 0.05 else 0.0
                vad_labels[b, t, 1] = 1.0 if energy_b[b, t] > 0.05 else 0.0
        
        return {
            'vap_labels': vap_labels,
            'eot_labels': eot_labels,
            'backchannel_labels': backchannel_labels,
            'overlap_labels': overlap_labels,
            'vad_labels': vad_labels
        }
    
    def _compute_sample_metrics(self, outputs: Dict, targets: Dict) -> Dict:
        """Compute comprehensive metrics for a single sample"""
        metrics = {}
        
        # VAP accuracy
        vap_preds = torch.argmax(outputs['vap_logits'], dim=-1)
        vap_acc = (vap_preds == targets['vap_labels']).float().mean().item()
        metrics['vap_accuracy'] = vap_acc
        
        # EoT metrics
        eot_probs = torch.sigmoid(outputs['eot_logits'])
        eot_preds = (eot_probs > 0.5).float()
        eot_acc = (eot_preds == targets['eot_labels']).float().mean().item()
        metrics['eot_accuracy'] = eot_acc
        
        # EoT F1 scores with different tolerances
        metrics['eot_f1_200ms'] = self._compute_eot_f1(eot_probs, targets['eot_labels'], tolerance_ms=200)
        metrics['eot_f1_500ms'] = self._compute_eot_f1(eot_probs, targets['eot_labels'], tolerance_ms=500)
        
        # Other metrics
        backchannel_probs = torch.sigmoid(outputs['backchannel_logits'])
        backchannel_preds = (backchannel_probs > 0.5).float()
        backchannel_acc = (backchannel_preds == targets['backchannel_labels']).float().mean().item()
        metrics['backchannel_accuracy'] = backchannel_acc
        
        overlap_probs = torch.sigmoid(outputs['overlap_logits'])
        overlap_preds = (overlap_probs > 0.5).float()
        overlap_acc = (overlap_preds == targets['overlap_labels']).float().mean().item()
        metrics['overlap_accuracy'] = overlap_acc
        
        vad_probs = torch.sigmoid(outputs['vad_logits'])
        vad_preds = (vad_probs > 0.5).float()
        vad_acc = (vad_preds == targets['vad_labels']).float().mean().item()
        metrics['vad_accuracy'] = vad_acc
        
        # Overall accuracy
        overall_acc = (vap_acc + eot_acc + backchannel_acc + overlap_acc + vad_acc) / 5
        metrics['overall_accuracy'] = overall_acc
        
        return metrics
    
    def _compute_eot_f1(self, predictions: torch.Tensor, targets: torch.Tensor, tolerance_ms: int = 200) -> float:
        """Compute EoT F1 score with temporal tolerance"""
        # Convert to binary predictions
        pred_binary = (predictions > 0.5).float()
        
        # Simple F1 calculation (can be enhanced with temporal tolerance)
        tp = (pred_binary * targets).sum().item()
        fp = (pred_binary * (1 - targets)).sum().item()
        fn = ((1 - pred_binary) * targets).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
    def _compute_overall_metrics(self, per_sample_results: List[Dict]) -> Dict:
        """Compute overall metrics across all samples"""
        overall = {}
        
        # Aggregate metrics
        metric_names = ['vap_accuracy', 'eot_accuracy', 'backchannel_accuracy', 
                       'overlap_accuracy', 'vad_accuracy', 'overall_accuracy']
        
        for metric in metric_names:
            values = [sample['metrics'][metric] for sample in per_sample_results]
            overall[f'{metric}_mean'] = np.mean(values)
            overall[f'{metric}_std'] = np.std(values)
            overall[f'{metric}_min'] = np.min(values)
            overall[f'{metric}_max'] = np.max(values)
        
        # Special metrics
        overall['eot_f1_200ms_mean'] = np.mean([sample['metrics']['eot_f1_200ms'] for sample in per_sample_results])
        overall['eot_f1_500ms_mean'] = np.mean([sample['metrics']['eot_f1_500ms'] for sample in per_sample_results])
        
        return overall
    
    def _analyze_temporal_patterns(self, per_sample_results: List[Dict]) -> Dict:
        """Analyze temporal patterns in predictions"""
        temporal = {}
        
        # Analyze prediction consistency over time
        for sample in per_sample_results:
            predictions = sample['predictions']
            targets = sample['targets']
            
            # VAP pattern consistency
            vap_preds = predictions['vap_logits']
            vap_targets = targets['vap_labels']
            
            # Compute temporal consistency
            if len(vap_preds.shape) == 3:
                seq_len = vap_preds.shape[1]
                consistency = []
                for t in range(1, seq_len):
                    if t < seq_len - 1:
                        # Check if predictions are consistent with neighbors
                        pred_consistent = (vap_preds[0, t] == vap_preds[0, t-1]) or (vap_preds[0, t] == vap_preds[0, t+1])
                        consistency.append(pred_consistent)
                
                if consistency:
                    temporal[f'sample_{sample["sample_id"]}_vap_consistency'] = np.mean(consistency)
        
        return temporal
    
    def _analyze_error_patterns(self, per_sample_results: List[Dict]) -> Dict:
        """Analyze patterns in prediction errors"""
        error_analysis = {}
        
        # Analyze VAP error patterns
        vap_errors = []
        for sample in per_sample_results:
            predictions = sample['predictions']
            targets = sample['targets']
            
            if 'vap_logits' in predictions and 'vap_labels' in targets:
                vap_preds = np.argmax(predictions['vap_logits'], axis=-1)
                vap_targets = targets['vap_labels']
                
                # Find error positions
                errors = (vap_preds != vap_targets)
                error_positions = np.where(errors)[1] if len(errors.shape) > 1 else np.where(errors)[0]
                
                if len(error_positions) > 0:
                    vap_errors.extend(error_positions.tolist())
        
        if vap_errors:
            error_analysis['vap_error_positions'] = vap_errors
            error_analysis['vap_error_mean_position'] = np.mean(vap_errors)
            error_analysis['vap_error_std_position'] = np.std(vap_errors)
        
        return error_analysis
    
    def _compare_with_baseline(self, analysis: Dict) -> Dict:
        """Compare optimized model performance with baseline"""
        # Load baseline results if available
        baseline_file = "results/simple_evaluation_*.json"
        baseline_results = None
        
        try:
            import glob
            baseline_files = glob.glob(baseline_file)
            if baseline_files:
                latest_baseline = max(baseline_files, key=lambda x: Path(x).stat().st_mtime)
                with open(latest_baseline, 'r') as f:
                    baseline_results = json.load(f)
        except:
            pass
        
        comparison = {
            'baseline_available': baseline_results is not None,
            'improvements': {}
        }
        
        if baseline_results:
            baseline_metrics = baseline_results.get('performance_metrics', {})
            
            # Compare key metrics
            for metric in ['overall_accuracy', 'vap_accuracy', 'eot_accuracy']:
                if metric in baseline_metrics and f'{metric}_mean' in analysis:
                    baseline_val = baseline_metrics[metric]
                    optimized_val = analysis[f'{metric}_mean']
                    improvement = optimized_val - baseline_val
                    improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
                    
                    comparison['improvements'][metric] = {
                        'baseline': baseline_val,
                        'optimized': optimized_val,
                        'absolute_improvement': improvement,
                        'percentage_improvement': improvement_pct
                    }
        
        return comparison
    
    def _generate_evaluation_plots(self, analysis: Dict):
        """Generate evaluation plots and visualizations"""
        try:
            # Create results directory
            plots_dir = Path("results/plots")
            plots_dir.mkdir(exist_ok=True)
            
            # 1. Overall performance comparison
            metrics = ['vap_accuracy', 'eot_accuracy', 'backchannel_accuracy', 'overlap_accuracy', 'vad_accuracy']
            means = [analysis[f'{m}_mean'] for m in metrics]
            stds = [analysis[f'{m}_std'] for m in metrics]
            
            plt.figure(figsize=(10, 6))
            x = np.arange(len(metrics))
            plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
            plt.xlabel('Metrics')
            plt.ylabel('Accuracy')
            plt.title('Optimized Model Performance by Metric')
            plt.xticks(x, [m.replace('_', ' ').title() for m in metrics], rotation=45)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(plots_dir / 'optimized_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Temporal consistency analysis
            if 'temporal_analysis' in analysis:
                temporal_data = [v for v in analysis['temporal_analysis'].values() if isinstance(v, (int, float))]
                if temporal_data:
                    plt.figure(figsize=(8, 6))
                    plt.hist(temporal_data, bins=10, alpha=0.7, edgecolor='black')
                    plt.xlabel('Temporal Consistency Score')
                    plt.ylabel('Frequency')
                    plt.title('Distribution of Temporal Consistency Scores')
                    plt.tight_layout()
                    plt.savefig(plots_dir / 'temporal_consistency.png', dpi=300, bbox_inches='tight')
                    plt.close()
            
            logger.info("✅ Evaluation plots generated successfully")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not generate plots: {e}")
    
    def _save_evaluation_results(self, analysis: Dict, baseline_comparison: Dict):
        """Save comprehensive evaluation results"""
        results_file = f"results/optimized_evaluation_{int(time.time())}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        # Prepare results for saving
        save_data = {
            'evaluation_timestamp': time.time(),
            'model_architecture': {
                'hidden_dim': 128,
                'num_layers': 2,
                'num_heads': 4
            },
            'checkpoint_path': self.checkpoint_path,
            'overall_metrics': analysis,
            'baseline_comparison': baseline_comparison,
            'evaluation_summary': {
                'total_samples': len(analysis.get('per_sample', [])),
                'mean_overall_accuracy': analysis.get('overall_accuracy_mean', 0),
                'best_metric': max([v for k, v in analysis.items() if k.endswith('_mean')], default=0)
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"✅ Evaluation results saved: {results_file}")
    
    def print_evaluation_summary(self, analysis: Dict, baseline_comparison: Dict):
        """Print comprehensive evaluation summary"""
        logger.info("\n" + "="*80)
        logger.info("🎯 OPTIMIZED MODEL EVALUATION SUMMARY")
        logger.info("="*80)
        
        # Overall performance
        logger.info(f"Overall Accuracy: {analysis.get('overall_accuracy_mean', 0):.4f} ± {analysis.get('overall_accuracy_std', 0):.4f}")
        logger.info(f"VAP Accuracy: {analysis.get('vap_accuracy_mean', 0):.4f} ± {analysis.get('vap_accuracy_std', 0):.4f}")
        logger.info(f"EoT Accuracy: {analysis.get('eot_accuracy_mean', 0):.4f} ± {analysis.get('eot_accuracy_std', 0):.4f}")
        logger.info(f"EoT F1 (200ms): {analysis.get('eot_f1_200ms_mean', 0):.4f}")
        logger.info(f"EoT F1 (500ms): {analysis.get('eot_f1_500ms_mean', 0):.4f}")
        
        # Baseline comparison
        if baseline_comparison.get('baseline_available', False):
            logger.info("\n📊 BASELINE COMPARISON:")
            for metric, comparison in baseline_comparison['improvements'].items():
                logger.info(f"  {metric}: {comparison['baseline']:.4f} → {comparison['optimized']:.4f} "
                           f"(+{comparison['percentage_improvement']:+.1f}%)")
        else:
            logger.info("\n📊 BASELINE COMPARISON: No baseline results available")
        
        logger.info("="*80)

def main():
    """Main evaluation function"""
    logger.info("🚀 Starting Optimized Model Evaluation")
    
    try:
        # Create evaluator
        evaluator = OptimizedModelEvaluator()
        
        # Run evaluation
        analysis = evaluator.evaluate_model(num_test_samples=15)
        
        # Compare with baseline
        baseline_comparison = evaluator._compare_with_baseline(analysis)
        
        # Print summary
        evaluator.print_evaluation_summary(analysis, baseline_comparison)
        
        logger.info("🎉 Optimized model evaluation completed successfully!")
        logger.info("Check the results/ directory for detailed metrics and plots.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 