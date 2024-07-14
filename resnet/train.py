import os, random, json
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from loguru import logger
from tqdm import tqdm

class PokemonDataset(Dataset):
    def __init__(self, data_split, classes):
        self.classes_mapping = {c: i for i, c in enumerate(classes)}
        self.images = []
        self.classes = []
        for img_path,img_class in data_split:
            if img_path.endswith('.svg'):
                # print("Skipping",img_path)
                continue
            self.images.append(img_path) # 只记录路径
            self.classes.append(self.classes_mapping[img_class]) # 记录对应的idx

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 因为PNG是4通道的，在preprocess时会有问题，所以需要convert到RGB
        return Image.open(self.images[idx]).convert("RGB"),self.classes[idx]

def collate_fn(batch, preprocess):
    images, labels = zip(*batch)
    images = [preprocess(img) for img in images]
    labels = torch.LongTensor(labels)
    images = torch.stack(images, dim=0) # (N, C, H, W) = (32, 3, 224, 224)
    return images,labels

def split_data(data_path):
    classes = sorted([fn for fn in os.listdir(data_path) if ".DS_Store" not in fn and not fn.startswith(".")])
    cnt = {}
    for c in classes:
        cnt[c] = sorted([fn for fn in os.listdir(os.path.join(f"{data_path}",f"{c}"))  if ".DS_Store" not in fn and not fn.startswith(".")])

    # 每类里抽5张作为test
    # train,test : [[img_path, img_class], ...]
    train,test = [],[]
    for c in classes:
        test_filenames = random.sample(cnt[c],5)
        for i,filename in enumerate(cnt[c]):
            img_path = os.path.join(data_path,c,filename)
            img_class = c
            if filename in test_filenames:
                test.append([img_path,img_class])
            else:
                train.append([img_path,img_class])
    return train, test, classes

def calculate_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    logger.info("Evaluating on test ...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(model.device)
            labels = labels.to(model.device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure that CUDA deterministic algorithms are used for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))       # 项目根目录文件夹
    DATA_PATH = os.path.join(f"{ROOT_DIR}","data","PokemonData")
    MODEL_PATH =os.path.join(f"{ROOT_DIR}","models","resnet50-bs=128-fc-only") # 存放模型权重的地方
    EPOCH = 10
    BATCH_SIZE = 128
    LOG_FREQ = int(3000/BATCH_SIZE)  # 多少个step记录1次log
    FC_ONLY = True # 是否只训练最顶层的fc

    os.makedirs(MODEL_PATH, exist_ok=True)
    logger.add(os.path.join(MODEL_PATH,"training_log.log"), format="{time} {level} {message}", level="INFO")
    set_seed(42)

    # 保存train_split 和 test_split
    train_split, test_split, classes = split_data(DATA_PATH)  # 6087, 750
    # json.dump(train_split,open(f"train_split.json","w"), indent=True)
    # json.dump(test_split, open(f"test_split.json", "w"), indent=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    model = resnet50(weights=weights)
    hidden_dim = model.fc.weight.shape[1] # (1000, 2048)
    model.fc = torch.nn.Linear(hidden_dim, len(classes)) # 重新初始化一个Linear层
    model.device = device
    # 是否只训练最顶上的分类层
    for k, v in model.named_parameters():
        if FC_ONLY and not k.startswith("fc."):
            v.requires_grad = False
    model.to(device)
    optimizer = Adam(model.parameters(), lr=5e-4)
    loss_fn = CrossEntropyLoss()
    logger.info(f"Finished loading models, only train the fc = {FC_ONLY}, device = {device}")


    # 实际上会抛弃一些svg图像
    train_dataset = PokemonDataset(train_split, classes)
    test_dataset = PokemonDataset(test_split, classes)
    logger.info("Train",len(train_dataset),"Test",len(test_dataset))
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=lambda x: collate_fn(x,preprocess))
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=lambda x: collate_fn(x,preprocess))

    best_test_acc = -1
    step = 0
    for epoch in range(1,EPOCH+1):
        epoch_loss = 0.0

        for (images, labels) in tqdm(train_loader):
            images = images.to(model.device)
            labels = labels.to(model.device)

            model.train()
            optimizer.zero_grad()
            output = model(images)  # (N, #classes)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            step += 1
            epoch_loss += loss.item() * images.shape[0]
            if step % LOG_FREQ == 0:
                accuracy = calculate_accuracy(model, test_loader)
                logger.info(f"Epoch:{epoch}: Step {step}: Training Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.2f}%")
                if accuracy>best_test_acc:
                    best_test_acc = accuracy
                    print("save to", os.path.join(f"{MODEL_PATH}",f"e={epoch}-s={step}-acc={accuracy:.2f}.pth"))
                    torch.save(model.state_dict(), os.path.join(f"{MODEL_PATH}",f"e={epoch}-s={step}-acc={accuracy:.2f}.pth"))

        epoch_loss /= len(train_dataset)

        accuracy = calculate_accuracy(model, test_loader)
        logger.info(f"Epoch:{epoch}, Epoch Loss {epoch_loss:.2f}, Test Accuracy {accuracy:.2f}%")
        if accuracy > best_test_acc:
            best_test_acc = accuracy
            torch.save(model.state_dict(), os.path.join(f"{MODEL_PATH}",f"e={epoch}-s={step}-acc={accuracy:.2f}.pth"))