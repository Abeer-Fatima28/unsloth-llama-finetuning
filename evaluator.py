# import json
# import numpy as np
# from collections import Counter, defaultdict
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
# from tqdm import tqdm

# class DocumentEvaluator:
#     def __init__(self):
#         self.classes = set()
#         self.predictions = []
#         self.ground_truth = []
#         self.document_level_metrics = {}
#         self.class_level_metrics = {}
#         self.confusion_matrices = {}

#     def parse_label(self, label_str):
#         """Parse label JSON string to dictionary."""
#         if isinstance(label_str, str):
#             try:
#                 return json.loads(label_str)
#             except json.JSONDecodeError:
#                 print(f"Error parsing JSON: {label_str}")
#                 return {}
#         return label_str

#     def collect_classes(self):
#         """Collect all unique class names from predictions and ground truth."""
#         for pred, truth in zip(self.predictions, self.ground_truth):
#             for cls in list(pred.keys()) + list(truth.keys()):
#                 self.classes.add(cls)
#         self.classes = sorted(list(self.classes))
#         print(f"Found {len(self.classes)} unique classes: {self.classes}")

#     def add_sample(self, prediction, ground_truth):
#         """Add a prediction and ground truth pair for evaluation."""
#         pred_dict = self.parse_label(prediction)
#         truth_dict = self.parse_label(ground_truth)

#         self.predictions.append(pred_dict)
#         self.ground_truth.append(truth_dict)

#     def evaluate_document_level(self):
#         """Calculate document-level metrics."""
#         exact_matches = 0
#         partial_matches = 0
#         no_matches = 0  # New category for completely different predictions
        
#         # For tracking match quality
#         overlap_scores = []  # Will store the Jaccard similarity for each document
        
#         # For macro averaging
#         doc_precision_sum = 0
#         doc_recall_sum = 0
#         doc_f1_sum = 0
    
#         # For each document
#         for pred, truth in zip(self.predictions, self.ground_truth):
#             pred_keys = set(pred.keys())
#             truth_keys = set(truth.keys())
            
#             # Calculate overlap using Jaccard similarity
#             intersection = pred_keys.intersection(truth_keys)
#             union = pred_keys.union(truth_keys)
#             overlap_score = len(intersection) / len(union) if len(union) > 0 else 1.0
#             overlap_scores.append(overlap_score)
    
#             if pred_keys == truth_keys:
#                 # Check if page indices match exactly for each class
#                 pages_match = True
#                 for key in pred_keys:
#                     if sorted(pred[key]) != sorted(truth[key]):
#                         pages_match = False
#                         break
    
#                 if pages_match:
#                     exact_matches += 1
#                 else:
#                     partial_matches += 1
#             elif len(intersection) > 0:
#                 # Some overlap but not exact match
#                 partial_matches += 1
#             else:
#                 # No overlap at all
#                 no_matches += 1
    
#             # Calculate precision, recall, F1 for this document
#             if len(pred_keys) == 0 and len(truth_keys) == 0:
#                 precision = 1.0
#                 recall = 1.0
#             elif len(pred_keys) == 0:
#                 precision = 0.0
#                 recall = 0.0
#             elif len(truth_keys) == 0:
#                 precision = 0.0
#                 recall = 0.0
#             else:
#                 precision = len(intersection) / len(pred_keys)
#                 recall = len(intersection) / len(truth_keys)
    
#             f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
#             doc_precision_sum += precision
#             doc_recall_sum += recall
#             doc_f1_sum += f1
    
#         # Calculate macro averages
#         num_docs = len(self.predictions)
#         macro_precision = doc_precision_sum / num_docs
#         macro_recall = doc_recall_sum / num_docs
#         macro_f1 = doc_f1_sum / num_docs
#         avg_overlap = np.mean(overlap_scores) if overlap_scores else 0
    
#         # Store document-level metrics
#         self.document_level_metrics = {
#             'exact_match_rate': exact_matches / num_docs,
#             'partial_match_rate': partial_matches / num_docs,
#             'no_match_rate': no_matches / num_docs,  # New metric
#             'average_overlap': avg_overlap,  # New metric showing average Jaccard similarity
#             'macro_precision': macro_precision,
#             'macro_recall': macro_recall,
#             'macro_f1': macro_f1,
#             'num_samples': num_docs
#         }
    
#     # def evaluate_document_level(self):
#     #     """Calculate document-level metrics."""
#     #     exact_matches = 0
#     #     partial_matches = 0

#     #     # For macro averaging
#     #     doc_precision_sum = 0
#     #     doc_recall_sum = 0
#     #     doc_f1_sum = 0

#     #     # For each document
#     #     for pred, truth in zip(self.predictions, self.ground_truth):
#     #         pred_keys = set(pred.keys())
#     #         truth_keys = set(truth.keys())

#     #         if pred_keys == truth_keys:
#     #             # Check if page indices match exactly for each class
#     #             pages_match = True
#     #             for key in pred_keys:
#     #                 if sorted(pred[key]) != sorted(truth[key]):
#     #                     pages_match = False
#     #                     break

#     #             if pages_match:
#     #                 exact_matches += 1
#     #             else:
#     #                 partial_matches += 1
#     #         else:
#     #             partial_matches += 1

#     #         # Calculate precision, recall, F1 for this document
#     #         if len(pred_keys) == 0 and len(truth_keys) == 0:
#     #             precision = 1.0
#     #             recall = 1.0
#     #         elif len(pred_keys) == 0:
#     #             precision = 0.0
#     #             recall = 0.0
#     #         elif len(truth_keys) == 0:
#     #             precision = 0.0
#     #             recall = 0.0
#     #         else:
#     #             precision = len(pred_keys.intersection(truth_keys)) / len(pred_keys)
#     #             recall = len(pred_keys.intersection(truth_keys)) / len(truth_keys)

#     #         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

#     #         doc_precision_sum += precision
#     #         doc_recall_sum += recall
#     #         doc_f1_sum += f1

#     #     # Calculate macro averages
#     #     num_docs = len(self.predictions)
#     #     macro_precision = doc_precision_sum / num_docs
#     #     macro_recall = doc_recall_sum / num_docs
#     #     macro_f1 = doc_f1_sum / num_docs

#     #     # Store document-level metrics
#     #     self.document_level_metrics = {
#     #         'exact_match_rate': exact_matches / num_docs,
#     #         'partial_match_rate': partial_matches / num_docs,
#     #         'macro_precision': macro_precision,
#     #         'macro_recall': macro_recall,
#     #         'macro_f1': macro_f1,
#     #         'num_samples': num_docs
#     #     }

#     def evaluate_class_level(self):
#         """Calculate class-level metrics."""
#         # Convert to binary classification problem for each class
#         class_preds = {cls: [] for cls in self.classes}
#         class_truths = {cls: [] for cls in self.classes}

#         for pred, truth in zip(self.predictions, self.ground_truth):
#             for cls in self.classes:
#                 class_preds[cls].append(1 if cls in pred else 0)
#                 class_truths[cls].append(1 if cls in truth else 0)

#         # Calculate metrics for each class
#         for cls in self.classes:
#             precision, recall, f1, _ = precision_recall_fscore_support(
#                 class_truths[cls], class_preds[cls], average='binary')

#             accuracy = accuracy_score(class_truths[cls], class_preds[cls])

#             tn, fp, fn, tp = confusion_matrix(class_truths[cls], class_preds[cls]).ravel()

#             self.class_level_metrics[cls] = {
#                 'precision': precision,
#                 'recall': recall,
#                 'f1': f1,
#                 'accuracy': accuracy,
#                 'support': sum(class_truths[cls]),
#                 'true_positives': tp,
#                 'false_positives': fp,
#                 'true_negatives': tn,
#                 'false_negatives': fn
#             }

#             self.confusion_matrices[cls] = {
#                 'true_positive': tp,
#                 'false_positive': fp,
#                 'true_negative': tn,
#                 'false_negative': fn
#             }

#     def evaluate_page_level(self):
#         """Calculate page-level accuracy for each class."""
#         page_metrics = {}

#         for cls in self.classes:
#             page_correct = 0
#             page_total = 0

#             page_true_positives = 0
#             page_false_positives = 0
#             page_false_negatives = 0

#             for pred, truth in zip(self.predictions, self.ground_truth):
#                 # Get page indices for this class
#                 pred_pages = set()
#                 if cls in pred:
#                     for page_group in pred[cls]:
#                         if isinstance(page_group, list):
#                             pred_pages.update(page_group)
#                         else:
#                             pred_pages.add(page_group)

#                 truth_pages = set()
#                 if cls in truth:
#                     for page_group in truth[cls]:
#                         if isinstance(page_group, list):
#                             truth_pages.update(page_group)
#                         else:
#                             truth_pages.add(page_group)

#                 # Calculate metrics
#                 for page in pred_pages:
#                     if page in truth_pages:
#                         page_true_positives += 1
#                     else:
#                         page_false_positives += 1

#                 page_false_negatives += len(truth_pages - pred_pages)

#             # Calculate precision, recall, F1
#             if page_true_positives + page_false_positives == 0:
#                 precision = 0
#             else:
#                 precision = page_true_positives / (page_true_positives + page_false_positives)

#             if page_true_positives + page_false_negatives == 0:
#                 recall = 0
#             else:
#                 recall = page_true_positives / (page_true_positives + page_false_negatives)

#             if precision + recall == 0:
#                 f1 = 0
#             else:
#                 f1 = 2 * precision * recall / (precision + recall)

#             page_metrics[cls] = {
#                 'precision': precision,
#                 'recall': recall,
#                 'f1': f1,
#                 'true_positives': page_true_positives,
#                 'false_positives': page_false_positives,
#                 'false_negatives': page_false_negatives
#             }

#         return page_metrics

#     def evaluate(self):
#         """Run all evaluations and return results."""
#         if not self.predictions or not self.ground_truth:
#             raise ValueError("No samples to evaluate. Add samples first.")

#         # Collect all unique classes
#         self.collect_classes()

#         # Run evaluations
#         self.evaluate_document_level()
#         self.evaluate_class_level()
#         page_metrics = self.evaluate_page_level()

#         # Calculate micro and macro averages for class-level metrics
#         class_precisions = [metrics['precision'] for metrics in self.class_level_metrics.values()]
#         class_recalls = [metrics['recall'] for metrics in self.class_level_metrics.values()]
#         class_f1s = [metrics['f1'] for metrics in self.class_level_metrics.values()]

#         macro_precision = np.mean(class_precisions)
#         macro_recall = np.mean(class_recalls)
#         macro_f1 = np.mean(class_f1s)

#         # Format results
#         results = {
#             'document_level': self.document_level_metrics,
#             'class_level': self.class_level_metrics,
#             'page_level': page_metrics,
#             'macro_averages': {
#                 'precision': macro_precision,
#                 'recall': macro_recall,
#                 'f1': macro_f1
#             },
#             'confusion_matrices': self.confusion_matrices,
#             'num_classes': len(self.classes),
#             'class_names': self.classes
#         }

#         return results

#     def print_results(self, results):
#         """Print evaluation results in a readable format."""
#         print("\n===== DOCUMENT CLASSIFICATION EVALUATION RESULTS =====")

#         print("\n----- Document-Level Metrics -----")
#         print(f"Number of documents: {results['document_level']['num_samples']}")
#         print(f"Exact match rate: {results['document_level']['exact_match_rate']:.4f}")
#         print(f"Partial match rate: {results['document_level']['partial_match_rate']:.4f}")
#         print(f"Macro Precision: {results['document_level']['macro_precision']:.4f}")
#         print(f"Macro Recall: {results['document_level']['macro_recall']:.4f}")
#         print(f"Macro F1: {results['document_level']['macro_f1']:.4f}")

#         print("\n----- Class-Level Metrics -----")
#         print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
#         print("-" * 65)

#         for cls in sorted(results['class_level'].keys()):
#             metrics = results['class_level'][cls]
#             print(f"{cls:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['support']:<10}")

#         print("\n----- Macro Averages -----")
#         print(f"Macro Precision: {results['macro_averages']['precision']:.4f}")
#         print(f"Macro Recall: {results['macro_averages']['recall']:.4f}")
#         print(f"Macro F1: {results['macro_averages']['f1']:.4f}")

#         print("\n----- Confusion Matrices -----")
#         for cls in sorted(results['confusion_matrices'].keys()):
#             cm = results['confusion_matrices'][cls]
#             print(f"\n{cls}:")
#             print(f"True Positives: {cm['true_positive']}")
#             print(f"False Positives: {cm['false_positive']}")
#             print(f"True Negatives: {cm['true_negative']}")
#             print(f"False Negatives: {cm['false_negative']}")

#         print("\n===== END OF EVALUATION RESULTS =====")



import json
from typing import List, Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict, Counter


class DocumentEvaluator:
    def __init__(self):
        self.predictions = []
        self.ground_truths = []
        self.labels = set()

    def add_sample(self, pred_str: str, true_str: str):
        try:
            pred_json = json.loads(pred_str)
            true_json = json.loads(true_str)

            self.predictions.append(pred_json)
            self.ground_truths.append(true_json)

            self.labels.update(pred_json.keys())
            self.labels.update(true_json.keys())

        except json.JSONDecodeError:
            print("Error parsing JSON:", pred_str)

    def _get_binary_vectors(self, pred_dict, true_dict, label):
        pred_vector = [0] * 1000  # assuming max 1000 pages
        true_vector = [0] * 1000

        for group in pred_dict.get(label, []):
            for page in group:
                if isinstance(page, int) and 0 <= page < len(pred_vector):
                    pred_vector[page] = 1

        for group in true_dict.get(label, []):
            for page in group:
                if isinstance(page, int) and 0 <= page < len(true_vector):
                    true_vector[page] = 1

        return pred_vector, true_vector

    def evaluate(self):
        label_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        doc_exact_matches = 0
        partial_matches = 0
        total_docs = len(self.predictions)
        skipped = 0

        for pred_dict, true_dict in zip(self.predictions, self.ground_truths):
            if not isinstance(pred_dict, dict) or not isinstance(true_dict, dict):
                skipped += 1
                continue

            local_exact = True
            local_partial = False

            for label in self.labels:
                pred_vec, true_vec = self._get_binary_vectors(pred_dict, true_dict, label)

                tp = sum(p == t == 1 for p, t in zip(pred_vec, true_vec))
                fp = sum(p == 1 and t == 0 for p, t in zip(pred_vec, true_vec))
                fn = sum(p == 0 and t == 1 for p, t in zip(pred_vec, true_vec))

                label_metrics[label]["tp"] += tp
                label_metrics[label]["fp"] += fp
                label_metrics[label]["fn"] += fn

                if fp > 0 or fn > 0:
                    local_exact = False
                if tp > 0 and (fp > 0 or fn > 0):
                    local_partial = True

            if local_exact:
                doc_exact_matches += 1
            elif local_partial:
                partial_matches += 1

        macro_precision = sum(
            self._safe_divide(m["tp"], m["tp"] + m["fp"]) for m in label_metrics.values()
        ) / len(label_metrics)

        macro_recall = sum(
            self._safe_divide(m["tp"], m["tp"] + m["fn"]) for m in label_metrics.values()
        ) / len(label_metrics)

        macro_f1 = self._safe_divide(2 * macro_precision * macro_recall, macro_precision + macro_recall)

        print("\n===== DOCUMENT CLASSIFICATION EVALUATION RESULTS =====\n")
        print("----- Document-Level Metrics -----")
        print(f"Number of documents: {total_docs - skipped}")
        print(f"Exact match rate: {doc_exact_matches / total_docs:.4f}")
        print(f"Partial match rate: {partial_matches / total_docs:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro F1: {macro_f1:.4f}\n")

        print("----- Class-Level Metrics -----")
        print(f"{'Class':20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 65)
        for label, m in label_metrics.items():
            prec = self._safe_divide(m["tp"], m["tp"] + m["fp"])
            rec = self._safe_divide(m["tp"], m["tp"] + m["fn"])
            f1 = self._safe_divide(2 * prec * rec, prec + rec)
            support = m["tp"] + m["fn"]
            print(f"{label:20} {prec:10.4f} {rec:10.4f} {f1:10.4f} {support:10}")

        print("\n----- Macro Averages -----")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro F1: {macro_f1:.4f}\n")
        print("===== END OF EVALUATION RESULTS =====\n")
        return {
            "exact_match_rate": doc_exact_matches / total_docs,
            "partial_match_rate": partial_matches / total_docs,
            "no_match_rate": 1 - (doc_exact_matches + partial_matches) / total_docs,
            "average_overlap": macro_precision,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "num_samples": total_docs - skipped
        }

    def _safe_divide(self, num, denom):
        return num / denom if denom else 0.0


# ----------------- LEVEL 2 & 3 EVALUATION LOGIC -----------------

def normalize_grouped_pages(data: Dict[str, List[List[int]]]) -> Dict[str, List[Tuple[int]]]:
    """Convert list of page groups into sorted tuples for consistent comparison."""
    normalized = {}
    for doc_type, groups in data.items():
        if isinstance(groups, list):
            normalized[doc_type] = [tuple(sorted(g)) for g in groups if isinstance(g, list)]
    return normalized

def group_iou(g1: Tuple[int], g2: Tuple[int]) -> float:
    """Calculate intersection-over-union for two page groups."""
    set1, set2 = set(g1), set(g2)
    return len(set1 & set2) / len(set1 | set2)

def evaluate_strict_document_groups(preds: List[str], trues: List[str], iou_threshold: float = 1.0):
    total_exact_matches = 0
    total_ground_truth_segments = 0 # Accumulates actual ground truth segments across all docs and types
    total_matched_segments = 0      # Accumulates unique matched ground truth segments across all docs and types
    
    # These will accumulate total 'over' and 'under' segments across all documents
    total_over_segments_discrepancy = 0
    total_under_segments_discrepancy = 0

    valid_doc_count = 0

    for pred_str, true_str in zip(preds, trues):
        try:
            pred_dict = json.loads(pred_str)
            true_dict = json.loads(true_str)
        except json.JSONDecodeError:
            continue

        pred_norm = normalize_grouped_pages(pred_dict)
        true_norm = normalize_grouped_pages(true_dict)
        valid_doc_count += 1

        doc_exact_match_for_groups = True # Flag for Level 2 (document-wide exact grouping)
        current_doc_total_gt_segments = 0 # Total GT segments for *this document*
        current_doc_total_pred_segments = 0 # Total Pred segments for *this document*
        current_doc_matched_gt_segments = 0 # Total matched GT segments for *this document*

        all_doc_types_in_this_doc = set(pred_norm.keys()).union(true_norm.keys())

        for doc_type in all_doc_types_in_this_doc:
            pred_groups_for_type = list(pred_norm.get(doc_type, []))
            true_groups_for_type = list(true_norm.get(doc_type, []))

            current_doc_total_gt_segments += len(true_groups_for_type)
            current_doc_total_pred_segments += len(pred_groups_for_type)

            matched_true_indices_for_type = set() # Indices of GT segments matched in *this doc_type*
            used_pred_indices_for_type = set()   # Indices of Pred segments used in *this doc_type*

            # Greedy matching: for each true group, find its best predicted match
            for tg_idx, tg in enumerate(true_groups_for_type):
                best_iou = -1.0 # Initialize with -1.0 to ensure 0.0 is better
                best_pg_idx = -1

                for pg_idx, pg in enumerate(pred_groups_for_type):
                    if pg_idx in used_pred_indices_for_type:
                        continue # This predicted group is already used

                    iou = group_iou(pg, tg)
                    if iou > best_iou:
                        best_iou = iou
                        best_pg_idx = pg_idx

                # If a valid match is found (above threshold and a predicted group was identified)
                if best_iou >= iou_threshold and best_pg_idx != -1:
                    matched_true_indices_for_type.add(tg_idx)
                    used_pred_indices_for_type.add(best_pg_idx)

            # Accumulate matched GT segments for the current document
            current_doc_matched_gt_segments += len(matched_true_indices_for_type)

            # Check for exact match for Level 2
            # For Level 2, we need all groups for all doc_types to match exactly
            # This check applies per doc_type, if any doc_type doesn't match exactly, then doc_exact_match_for_groups becomes False
            if sorted(pred_groups_for_type) != sorted(true_groups_for_type):
                doc_exact_match_for_groups = False

            # Accumulate over/under segmentation discrepancies
            # These are *counts* of segments over/under for this specific doc_type
            total_over_segments_discrepancy += max(0, len(pred_groups_for_type) - len(used_pred_indices_for_type))
            total_under_segments_discrepancy += max(0, len(true_groups_for_type) - len(matched_true_indices_for_type))

        # After processing all doc_types for the current document, accumulate global totals
        total_ground_truth_segments += current_doc_total_gt_segments
        total_matched_segments += current_doc_matched_gt_segments # This is the crucial line for recall

        # Final check for Level 2 exact match at the document level
        if doc_exact_match_for_groups and \
           current_doc_total_gt_segments == current_doc_total_pred_segments: # Ensure total segment counts are also equal
            total_exact_matches += 1

    level2_accuracy = total_exact_matches / valid_doc_count if valid_doc_count else 0
    level3_segmentation_recall = total_matched_segments / total_ground_truth_segments if total_ground_truth_segments else 0
    
    # These rates are the average discrepancy per document (sum of discrepancies / number of valid documents)
    average_over_segments_per_doc = total_over_segments_discrepancy / valid_doc_count if valid_doc_count else 0
    average_under_segments_per_doc = total_under_segments_discrepancy / valid_doc_count if valid_doc_count else 0

    return {
        "level2_group_exact_match_rate": level2_accuracy,
        "level3_segmentation_recall": level3_segmentation_recall,
        "over_segmentation_rate": average_over_segments_per_doc,
        "under_segmentation_rate": average_under_segments_per_doc,
    }