# 基于transformers的实现
import os
from transformers import DetrConfig, DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
from train import DocLayNetDataset
import matplotlib.pyplot as plt

def classify(dataset, idx, model, processor):
    model.eval()

    if not (0<=idx<len(dataset)):
        print(f"Choose idx between 0 and {len(dataset)-1}")
        return

    data_item = dataset[idx]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=200)  # 1 row, 2 columns

    # 画 gt 对应的结果
    img = Image.open(data_item  ['image_path']).convert('RGB')
    # W, H = img.size
    draw = ImageDraw.Draw(img)
    for bbox, label_idx in zip(data_item['boxes'], data_item['class_labels']):
        # [xc, yc, h, w] -> [x1, y1, x2, y2]
        label = dataset.id2label[label_idx]
        color = dataset.label2color[label]
        bbox = dataset._center_corner(bbox)
        # 要乘以对应的比例
        bbox = [
            bbox[0] * data_item['W'],
            bbox[1] * data_item['H'],
            bbox[2] * data_item['W'],
            bbox[3] * data_item['H'],
        ]
        draw.rectangle(bbox, outline=color)
        # 这里展示的是左上角
        draw.text((bbox[0], bbox[1] - 10), text=label, fill=color)
    axes[0].imshow(img)
    axes[0].set_title("ground truth")
    axes[0].axis('off')  # Hide the axes

    # 展示模型预测结果
    img2 = Image.open(data_item['image_path']).convert('RGB')
    draw2 = ImageDraw.Draw(img2)

    # 模型推理, 只保留threshold=0.9的结果
    with torch.no_grad():
        inputs = processor(images=img2, return_tensors="pt")
        outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([img2.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
        bbox = [round(i, 2) for i in box.tolist()] # 这里已经转换到输入图像的尺寸了
        label = dataset.id2label[label_idx]
        color = dataset.label2color[label]
        draw2.rectangle(bbox, outline=color)
        # 展示label和对应的分数
        draw2.text((bbox[0], bbox[1] - 10), text=f"{label}-conf={score}", fill=color)

    axes[1].imshow(img2)
    axes[1].set_title("model prediction")
    axes[1].axis('off')  # Hide the axes

    # Display the figure
    plt.show()

if __name__=="__main__":
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(CUR_DIR)  # 项目根目录文件夹
    MODEL_PATH = os.path.join(f"{ROOT_DIR}", "models", "detr", "e=1-s=40-batch_loss=8.65.pth")
    val_dataset = DocLayNetDataset(os.path.join(ROOT_DIR, "data", "DocLayNet", "small_dataset", "val"))

    pretrained_config = DetrConfig.from_pretrained('facebook/detr-resnet-50', revision="no_timm")
    # 修改model.config
    # pretrained_config.num_queries = 200  # 如果要修改最大预测的bbox数目
    pretrained_config.label2id = val_dataset.label2id  # 新的id -> label
    pretrained_config.id2label = val_dataset.id2label

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DetrForObjectDetection(pretrained_config)
    model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
    model.to(device)
    processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50', revision="no_timm")
    model.eval()

    while True:
        idx = int(input("enter a index:\n"))
        classify(val_dataset, idx, model, processor)
        # ex: 12 Alakazam vs Kadabra
    print("Done!")
