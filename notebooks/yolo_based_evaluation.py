"""
YOLO-Based Evaluation on Test Set

This script treats the test set as if it were unlabeled (using YOLO for detection),
then evaluates the ML classifier's performance by comparing against ground truth labels.

This simulates real-world usage while allowing performance evaluation.

Usage:
    Add this code to the end of traditional_ml_classifiers.ipynb
"""

# ============================================================================
# Section 12: YOLO-Based Detection and Evaluation
# ============================================================================

print("="*70)
print("SECTION 12: YOLO-BASED DETECTION AND EVALUATION")
print("="*70)
print("\nThis section treats test images as unlabeled (using YOLO for detection),")
print("then evaluates performance by comparing against ground truth labels.\n")

# ----------------------------------------------------------------------------
# 12.1 Load YOLO Model
# ----------------------------------------------------------------------------

from ultralytics import YOLO

# Path to your trained YOLO model
# Update this path to your actual YOLO model
yolo_model_path = Path('../runs/detect/train/weights/best.pt')

if not yolo_model_path.exists():
    print(f"⚠️  YOLO model not found at: {yolo_model_path}")
    print(f"\nTo train a YOLO model:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('yolov8n.pt')")
    print(f"  model.train(data='../data/data.yaml', epochs=100)")
    print(f"\nSkipping YOLO-based evaluation...")
else:
    print(f"Loading YOLO model from: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    print("✓ YOLO model loaded successfully!\n")
    
    # ------------------------------------------------------------------------
    # 12.2 Define YOLO-Based Detection Function
    # ------------------------------------------------------------------------
    
    def detect_with_yolo_evaluate_with_labels(
        image_path: Path,
        yolo_model,
        ml_classifier,
        scaler,
        confidence_threshold=0.25
    ):
        """
        Detect algae using YOLO (as if unlabeled), then compare with ground truth.
        
        Returns:
            predictions: List of ML classifier predictions
            ground_truth: List of true labels
            detections: List of detection info
        """
        # Load image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Load ground truth labels for evaluation
        label_path = test_labels / (image_path.stem + '.txt')
        ground_truth_boxes = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        ground_truth_boxes.append({
                            'class_id': class_id,
                            'bbox': bbox
                        })
        
        # STAGE 1: Detect with YOLO (treating as unlabeled)
        yolo_results = yolo_model(image_path, conf=confidence_threshold, verbose=False)
        
        predictions = []
        detections = []
        
        for result in yolo_results:
            boxes = result.boxes
            
            for box in boxes:
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Convert to normalized format
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                box_w = (x2 - x1) / w
                box_h = (y2 - y1) / h
                bbox = [x_center, y_center, box_w, box_h]
                
                # STAGE 2: Extract features and classify
                features = extract_features_from_bbox(image_rgb, bbox)
                feature_vector = np.array([features[f] for f in selected_features]).reshape(1, -1)
                feature_vector_scaled = scaler.transform(feature_vector)
                
                # Predict with ML classifier
                pred_class = ml_classifier.predict(feature_vector_scaled)[0]
                
                if hasattr(ml_classifier, 'predict_proba'):
                    proba = ml_classifier.predict_proba(feature_vector_scaled)[0]
                    ml_confidence = proba[pred_class]
                else:
                    ml_confidence = 1.0
                
                yolo_confidence = float(box.conf[0])
                
                # Find matching ground truth (using IoU)
                matched_gt = None
                best_iou = 0.0
                
                for gt in ground_truth_boxes:
                    iou = calculate_iou(bbox, gt['bbox'])
                    if iou > best_iou and iou > 0.5:  # IoU threshold
                        best_iou = iou
                        matched_gt = gt
                
                detections.append({
                    'pred_class': pred_class,
                    'true_class': matched_gt['class_id'] if matched_gt else -1,
                    'ml_confidence': ml_confidence,
                    'yolo_confidence': yolo_confidence,
                    'iou': best_iou,
                    'bbox': bbox
                })
                
                if matched_gt:
                    predictions.append(pred_class)
                    # Note: We only evaluate on matched detections
        
        # Get ground truth labels for matched detections
        ground_truth = [d['true_class'] for d in detections if d['true_class'] != -1]
        matched_predictions = [d['pred_class'] for d in detections if d['true_class'] != -1]
        
        return matched_predictions, ground_truth, detections
    
    
    def calculate_iou(bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes."""
        # Convert from [x_center, y_center, w, h] to [x1, y1, x2, y2]
        def to_corners(bbox):
            x_c, y_c, w, h = bbox
            return [x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2]
        
        box1 = to_corners(bbox1)
        box2 = to_corners(bbox2)
        
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    
    print("✓ YOLO-based detection functions defined!\n")
    
    # ------------------------------------------------------------------------
    # 12.3 Run YOLO-Based Detection on Test Set
    # ------------------------------------------------------------------------
    
    print("Running YOLO-based detection on test set...")
    print("(Treating images as unlabeled, then comparing with ground truth)\n")
    
    all_predictions = []
    all_ground_truth = []
    all_detections = []
    
    test_image_files = sorted(list(test_images.glob('*.jpg')))
    
    for img_path in tqdm(test_image_files, desc="Processing test images"):
        preds, gt, dets = detect_with_yolo_evaluate_with_labels(
            img_path,
            yolo_model,
            best_model,
            scaler,
            confidence_threshold=0.25
        )
        
        all_predictions.extend(preds)
        all_ground_truth.extend(gt)
        all_detections.extend(dets)
    
    print(f"\nTotal detections: {len(all_detections)}")
    print(f"Matched detections (IoU > 0.5): {len(all_predictions)}")
    
    # ------------------------------------------------------------------------
    # 12.4 Evaluate Performance
    # ------------------------------------------------------------------------
    
    print("\n" + "="*70)
    print("YOLO-BASED EVALUATION RESULTS")
    print("="*70)
    
    if len(all_predictions) > 0:
        # Convert to numpy arrays
        y_pred_yolo = np.array(all_predictions)
        y_true_yolo = np.array(all_ground_truth)
        
        # Calculate metrics
        accuracy_yolo = accuracy_score(y_true_yolo, y_pred_yolo)
        precision_yolo = precision_score(y_true_yolo, y_pred_yolo, average='weighted', zero_division=0)
        recall_yolo = recall_score(y_true_yolo, y_pred_yolo, average='weighted', zero_division=0)
        f1_yolo = f1_score(y_true_yolo, y_pred_yolo, average='weighted', zero_division=0)
        
        print(f"\nPerformance Metrics (YOLO Detection + ML Classification):")
        print(f"  Accuracy:  {accuracy_yolo:.4f}")
        print(f"  Precision: {precision_yolo:.4f}")
        print(f"  Recall:    {recall_yolo:.4f}")
        print(f"  F1-Score:  {f1_yolo:.4f}")
        
        # Detection statistics
        avg_iou = np.mean([d['iou'] for d in all_detections if d['iou'] > 0])
        avg_yolo_conf = np.mean([d['yolo_confidence'] for d in all_detections])
        avg_ml_conf = np.mean([d['ml_confidence'] for d in all_detections])
        
        print(f"\nDetection Statistics:")
        print(f"  Average IoU (matched): {avg_iou:.4f}")
        print(f"  Average YOLO confidence: {avg_yolo_conf:.4f}")
        print(f"  Average ML confidence: {avg_ml_conf:.4f}")
        
        # ------------------------------------------------------------------------
        # 12.5 Confusion Matrix
        # ------------------------------------------------------------------------
        
        print("\nGenerating confusion matrix...")
        
        cm_yolo = confusion_matrix(y_true_yolo, y_pred_yolo)
        cm_yolo_normalized = cm_yolo.astype('float') / cm_yolo.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm_yolo_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Normalized Count'}
        )
        ax.set_title(f'{best_model_name} - YOLO-Based Detection\nConfusion Matrix', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(outputs_dir / 'confusion_matrix_yolo_based.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # ------------------------------------------------------------------------
        # 12.6 Classification Report
        # ------------------------------------------------------------------------
        
        print("\n" + "="*70)
        print("YOLO-BASED CLASSIFICATION REPORT")
        print("="*70)
        print(classification_report(y_true_yolo, y_pred_yolo, target_names=class_names, zero_division=0))
        
        # ------------------------------------------------------------------------
        # 12.7 Comparison with Ground Truth-Based Evaluation
        # ------------------------------------------------------------------------
        
        print("\n" + "="*70)
        print("COMPARISON: Ground Truth Labels vs YOLO Detection")
        print("="*70)
        
        comparison_table = pd.DataFrame({
            'Method': ['Ground Truth Labels', 'YOLO Detection'],
            'Accuracy': [best_accuracy, accuracy_yolo],
            'Precision': [
                precision_score(y_test, results[best_model_name]['predictions'], 
                               average='weighted', zero_division=0),
                precision_yolo
            ],
            'Recall': [
                recall_score(y_test, results[best_model_name]['predictions'], 
                            average='weighted', zero_division=0),
                recall_yolo
            ],
            'F1-Score': [
                f1_score(y_test, results[best_model_name]['predictions'], 
                        average='weighted', zero_division=0),
                f1_yolo
            ]
        })
        
        print("\n", comparison_table.to_string(index=False))
        
        # Visualize comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(comparison_table))
        width = 0.15
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width - 1.5*width, comparison_table[metric], 
                   width, label=metric, color=colors[i])
        
        ax.set_xlabel('Evaluation Method', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Performance Comparison: Ground Truth vs YOLO-Based Detection', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_table['Method'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(outputs_dir / 'comparison_gt_vs_yolo.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # ------------------------------------------------------------------------
        # 12.8 Summary
        # ------------------------------------------------------------------------
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        print(f"\n✓ Successfully evaluated {best_model_name} using YOLO-based detection")
        print(f"\nKey Findings:")
        print(f"  • Total test images: {len(test_image_files)}")
        print(f"  • Total YOLO detections: {len(all_detections)}")
        print(f"  • Matched detections (IoU > 0.5): {len(all_predictions)}")
        print(f"  • YOLO-based accuracy: {accuracy_yolo:.4f}")
        print(f"  • Ground truth-based accuracy: {best_accuracy:.4f}")
        
        accuracy_diff = best_accuracy - accuracy_yolo
        print(f"\n  • Performance difference: {accuracy_diff:.4f} ({accuracy_diff*100:.2f}%)")
        
        if accuracy_diff > 0:
            print(f"\n  ℹ️  The YOLO-based approach has slightly lower accuracy because:")
            print(f"     - YOLO may miss some algae (false negatives)")
            print(f"     - YOLO may detect false positives")
            print(f"     - Bounding box misalignment affects feature extraction")
        
        print(f"\n  ✓ This simulates real-world performance on unlabeled images!")
        
    else:
        print("\n⚠️  No matched detections found!")
        print("   This could mean:")
        print("   - YOLO model needs better training")
        print("   - Confidence threshold is too high")
        print("   - IoU threshold is too strict")
