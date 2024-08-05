# 基于transformers的实现
# https://github.com/roboflow/notebooks/blob/main/notebooks/train-huggingface-detr-on-custom-dataset.ipynb
import os, json, fitz, random
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from typing import List
from loguru import logger

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrConfig, DetrImageProcessor, DetrForObjectDetection
from torch.optim import AdamW
class DocLayNetDataset(Dataset):
    """class_labels: ..., boxes: [xc, yc, w, h], 预测的是一个[0,1]的比值"""
    def __init__(self, data_path, box_source = "box"):
        assert box_source in ["box", "box_line"]
        self.id2label = ["Caption", "Footnote", "Formula", "List-item",
                          "Page-footer", "Page-header", "Picture",
                          "Section-header", "Table", "Text", "Title"] # 一共11种类别
        self.label2id = {v:k for k,v in enumerate(self.id2label)}
        self.label2color = {'Caption': 'brown',
                             'Footnote': 'orange',
                             'Formula': 'gray',
                             'List-item': 'yellow',
                             'Page-footer': 'red',
                             'Page-header': 'red',
                             'Picture': 'violet',
                             'Section-header': 'orange',
                             'Table': 'green',
                             'Text': 'blue',
                             'Title': 'pink'
                            }
        self.data_path = data_path

        self.annotation_path = os.path.join(data_path, 'annotations')
        filenames = [fn for fn in os.listdir(self.annotation_path) if fn.endswith('.json') and not fn.startswith('.')]
        self.raw_data = []
        self.data = []
        for fn in tqdm(filenames):
            filepath = os.path.join(self.annotation_path, fn)
            page_annotation = json.load(open(filepath,"r")) # 一页的标注
            self.raw_data.append(page_annotation)
            # 每个元素有2个key: 'metadata', 'form'
            # ['metadata']['page_hash'] -> 图片的名字
            # form中的key都是'id_box', 'box', 'id_box_line', 'box_line', 'text', 'category', 'words', 'linking', 'font'
            # box是按照块划分的，box_line是按照行划分的，即切的比较碎。

            H, W = page_annotation['metadata']['original_height'], page_annotation['metadata']['original_width']
            H, W = int(H), int(W)
            H_coco, W_coco = page_annotation['metadata']['coco_height'], page_annotation['metadata']['coco_width']

            img_path = os.path.join(data_path, 'images_1x', page_annotation['metadata']['page_hash']+'.png')

            if not os.path.exists(img_path):
                # 创建对应的图片，因为images种的图默认是1025,1025的
                self._create_png(page_annotation['metadata']['page_hash'], W, H)

            # img = Image.open(img_path).convert('RGB')
            # assert (W,H)==img.size, f"metadata记录HW和原图对不上！{(W,H)}, {img.size}"

            visited = set()  # 不记录重复的bbox
            data_item = {"image_path": img_path, "class_labels": [], "boxes":[], "H": H, "W": W}
            for item in page_annotation['form']:
                bbox = item[box_source] # 'box' or 'box_line' -> 默认是按照coco_width标的
                bbox = [
                    np.clip((bbox[0] / W_coco),0,1),
                    np.clip((bbox[1] / H_coco), 0, 1),
                    np.clip((bbox[2] / W_coco), 0, 1),
                    np.clip((bbox[3] / H_coco), 0, 1)
                ]
                # 原始bbox数据就是用(left, top, width, height)标注的, 这里要转成[xc,yc,w,h]
                bbox = [bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, bbox[2], bbox[3]]
                if tuple(bbox) not in visited:
                    data_item["class_labels"].append(self.label2id[item['category']])
                    data_item["boxes"].append(bbox)
                    visited.add(tuple(bbox))
            self.data.append(data_item)
            assert len(data_item["class_labels"])==len(data_item["boxes"]) # 一个框对应1个boxes
        # 统计一张图片最多的bbox的个数
        print(f"Each image has MAXIMUM = {max(len(data_item['boxes']) for data_item in self.data)} bboxes")

    def _create_png(self, pdf_name, W, H):
        # 基于pdf得到对应的图片
        # assert 每个pdf只有1页
        pdf_path = os.path.join(self.data_path, 'pdfs', pdf_name+'.pdf')
        output_path = os.path.join(self.data_path, 'images_1x')
        os.makedirs(output_path, exist_ok=True)

        pdf_document = fitz.open(pdf_path)
        assert len(pdf_document)==1, "pdf超过1页"

        for page_number in range(len(pdf_document)):
            # Get the page
            page = pdf_document.load_page(page_number)

            zoom = 1.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            output_file = os.path.join(output_path,pdf_name+'.png')
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            resized_image = image.resize((W, H))
            resized_image.save(output_file)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _normalize(self, bbox: List[int], H: int, W: int):
        x1,y1,x2,y2 = bbox
        x1, x2 = x1 / W, x2 / W
        y1, y2 = y1 / H, y2 / H
        return [x1,y1,x2,y2]

    def _corner_center(self, bbox: List[float]):
        # 归一化的bbox[x1, y1, x2, y2] 变为[xc, yc, w, h]的表示
        x1, y1, x2, y2 = bbox
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return [xc, yc, w, h]

    def _center_corner(self, bbox: List[float]):
        # bbox [xc, yc, w, h] -> [x1, y1, x2, y2]
        xc, yc, w, h = bbox
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        return [x1, y1, x2, y2]

    def show(self, idx: int):
        assert  0<=idx<len(self.data), f"Chose idx from [0,{len(self.data)}]"
        data_item = self.data[idx]
        img_path = data_item['image_path']
        img = Image.open(img_path).convert('RGB')
        W,H = img.size

        assert (H,W) == (data_item['H'], data_item['W'])
        draw = ImageDraw.Draw(img)
        for bbox,label_idx in zip(data_item['boxes'],data_item['class_labels']):
            # [xc, yc, h, w] -> [x1, y1, x2, y2]
            label = self.id2label[label_idx]
            color = self.label2color[label]
            bbox = self._center_corner(bbox)
            # 要乘以对应的比例
            bbox = [
                bbox[0] * data_item['W'],
                bbox[1] * data_item['H'],
                bbox[2] * data_item['W'],
                bbox[3] * data_item['H'],
            ]
            draw.rectangle(bbox, outline=color)
            # 这里展示的是左上角
            draw.text((bbox[0], bbox[1]-10), text=label, fill=color)
        img.show()

def collate_fn(batch, device='cpu'):
    result = {
        "images": [],      # list of pil image
        "targets": [],     # list of dict
        # "class_labels": torch.LongTensor (list of bbox's labels)
        # "boxes": torch.tensor (list of bbox's labels)
    }
    for item in batch:
        result["images"].append(Image.open(item['image_path']).convert('RGB'))
        result["targets"].append({
            "class_labels": torch.LongTensor(item['class_labels']).to(device),
            "boxes": torch.tensor(item['boxes'], dtype=torch.float32).to(device),
        })
    return result["images"], result["targets"]

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure that CUDA deterministic algorithms are used for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(42)
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(CUR_DIR)  # 项目根目录文件夹
    MODEL_PATH = os.path.join(f"{ROOT_DIR}", "models", "detr")
    os.makedirs(MODEL_PATH, exist_ok=True)

    train_dataset = DocLayNetDataset(os.path.join(ROOT_DIR, "data", "DocLayNet", "base_dataset", "train"))
    val_dataset = DocLayNetDataset(os.path.join(ROOT_DIR, "data", "DocLayNet", "base_dataset", "val"))
    test_dataset = DocLayNetDataset(os.path.join(ROOT_DIR, "data", "DocLayNet", "base_dataset", "test"))
    # train_dataset.show(123)
    # val_dataset.show(42)
    # test_dataset.show(42)

    # 开训! 每个step是1个bs，每grad_accumulation更新1次
    bs = 8 # 8
    grad_accumulation = 8 # 实际上的bs = grad_accumulation * bs
    lr = 1e-4
    lr_backbone = 1e-5 # resnet的lr
    weight_decay = 1e-4
    EPOCH = 40
    LOG_FREQ = 5*grad_accumulation # 枚多少个bs*grad_accumulation log一次

    os.makedirs(MODEL_PATH, exist_ok=True)
    logger.add(os.path.join(MODEL_PATH, "training_log.log"), format="{time} {level} {message}", level="INFO")
    set_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=lambda x:collate_fn(x,device=device))
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, collate_fn=lambda x:collate_fn(x,device=device))
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, collate_fn=lambda x:collate_fn(x,device=device))

    # 为了使用已有的权重
    device = "cuda" if torch.cuda.is_available() else "cpu"
    '''
    pretrained_config = DetrConfig.from_pretrained('facebook/detr-resnet-50', revision="no_timm")
    # 修改model.config
    # pretrained_config.num_queries = 200  # 如果要修改最大预测的bbox数目
    pretrained_config.label2id = train_dataset.label2id # 新的id -> label
    pretrained_config.id2label = train_dataset.id2label

    model = DetrForObjectDetection(pretrained_config)
    '''
    from transformers import DetrForObjectDetection

    model = DetrForObjectDetection.from_pretrained(
        'facebook/detr-resnet-50',
        num_labels=len(train_dataset.label2id),
        id2label=train_dataset.id2label,
        label2id=train_dataset.label2id,
        ignore_mismatched_sizes=True, # 比如最后一层的权重, 一开始是92类，现在是12类
        revision="no_timm"
    )

    # DETR authors decided to use different learning rate for backbone
    # you can learn more about it here:
    # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
    # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr_backbone,
        },
    ]
    optimizer =AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
    model.to(device)
    processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50', revision="no_timm")


    best_test_acc = -1
    step = 0
    for epoch in range(1, EPOCH + 1):
        epoch_loss = 0.0
        num_samples = 0
        model.train()
        optimizer.zero_grad()

        for images, targets in train_loader:
            # list of pil image
            # list of dict, dict需要包含下面2个key
            # "class_labels": torch.LongTensor (list of bbox's labels)
            # "boxes": torch.tensor (list of bbox's labels)

            inputs = processor(images=images, return_tensors="pt").to(device)  # pixel_values, pixel_mask(全是1)
            outputs = model(**inputs, labels=targets)
            loss = outputs.loss
            loss.backward()

            step += 1
            if step % grad_accumulation==0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * len(images) # bs = len(images)
            num_samples += len(images)
            if step % LOG_FREQ == 0:
                logger.info(
                    f"Epoch:{epoch}: Step {step}: Training Loss: {loss.item():.4f}")
                # print("save to", os.path.join(f"{MODEL_PATH}", f"e={epoch}-s={step}-batch_loss={loss.item():.2f}.pth"))
                # torch.save(model.state_dict(),
                #            os.path.join(f"{MODEL_PATH}", f"e={epoch}-s={step}-batch_loss={loss.item():.2f}.pth"))
            if step % 375 == 0:
                # 每过3000个batch保存1次结果
                torch.save(model.state_dict(),
                           os.path.join(f"{MODEL_PATH}", f"e={epoch}-s={step}-batch_loss={loss.item():.2f}.pth"))
        if step % grad_accumulation != 0:
            # 如果没能整除 -> 单独更新1次。
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss /= num_samples

        logger.info(f"Epoch:{epoch}, Epoch Loss {epoch_loss:.2f}")
        torch.save(model.state_dict(), os.path.join(f"{MODEL_PATH}", f"e={epoch}-s={step}-epoch_loss={epoch_loss:.2f}.pth"))
