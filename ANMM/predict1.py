import os
import json
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score
from torchvision import transforms
from model import AlexNet
import torch.nn as nn

def specificity_score(y_true, y_pred, labels):
    """计算多分类特异度"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    fp = cm.sum(axis=0) - np.diag(cm)
    return tn / (tn + fp + 1e-7)

def per_class_metrics(y_true, y_pred, labels):
    """计算每个类别的评估指标"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n_classes = len(labels)
    metrics = {}
    for i, label in enumerate(labels):
        # 计算每个类别的TP, FP, FN, TN
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        # 正确计算各项指标
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)

        metrics[label] = {
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'Specificity': round(specificity, 4),
            'F1': round(f1, 4),
            'Accuracy': round(accuracy, 4)
        }
    return metrics


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 加载类别映射
    with open('./class_indices.json', 'r') as f:
        class_indict = json.load(f)
    class_labels = list(class_indict.values())

    # 初始化模型
    model = AlexNet(num_classes=len(class_labels)).to(device)
    model.load_state_dict(torch.load('./AlexNet.pth', map_location=device))
    model.eval()

    # 测试数据收集
    all_preds = []
    all_labels = []
    test_dir = r'D:/team/alexnet learning/deep-learning-for-image-processing-master-juanji-msa/data_set/danzao15(1)/test'

    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = data_transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img_tensor)
                    loss = criterion(output, torch.tensor([class_labels.index(class_name)]).to(device))
                    test_loss += loss.item()
                    total_samples += 1
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, pred_idx = torch.max(probabilities, 1)
                    confidence = confidence.item()

                # 记录预测结果和真实标签
                predicted_class = class_indict[str(pred_idx.item())]
                all_preds.append(predicted_class)
                all_labels.append(class_name)

                print(f"Image: {img_path}")
                print(f"True Label: {class_name}")
                print(f"Predicted Label: {predicted_class}")
                print(f"Confidence: {confidence:.4f}")
                print(f"Loss: {loss.item():.4f}\n")

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

    # 计算评估指标
    cm = confusion_matrix(all_labels, all_preds, labels=class_labels)
    metrics = {
        'Accuracy': accuracy_score(all_labels, all_preds),
        'Recall': recall_score(all_labels, all_preds, average='weighted'),
        'Precision': precision_score(all_labels, all_preds, average='weighted'),
        'F1': f1_score(all_labels, all_preds, average='weighted'),
        'Specificity': np.mean(specificity_score(all_labels, all_preds, class_labels)),
        'Test_Loss': test_loss / total_samples
    }

    # 计算每个类别的详细指标
    per_class_metrics_dict = per_class_metrics(all_labels, all_preds, class_labels)

    # 输出结果
    print("Confusion Matrix:\n", cm)
    print("\nOverall Evaluation Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nPer-class Evaluation Metrics:")
    for class_name, class_metrics in per_class_metrics_dict.items():
        print(f"\nClass: {class_name}")
        for metric_name, metric_value in class_metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")


if __name__ == '__main__':
    main()