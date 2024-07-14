import os, random
import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from train import split_data, set_seed
import matplotlib.pyplot as plt

def classify(data_split, idx, classes, model, preprocess):
    model.eval()
    classes_mapping = {c: i for i, c in enumerate(classes)}

    if not (0<=idx<len(data_split)):
        print(f"Choose idx between 0 and {len(data_split)-1}")
        return
    img = Image.open(data_split[idx][0]).convert('RGB')
    gt_class = data_split[idx][1]

    with torch.no_grad():
        processed_img = preprocess(img).unsqueeze(0).to(model.device)
        output = model(processed_img)
        pred = output.argmax(dim=1).item()

    # Set up the plot
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f'gt = {gt_class}, pred = {classes[pred]}')
    ax.axis('off')
    plt.show()


if __name__ == '__main__':
    DATA_PATH = "../data/PokemonData"
    MODEL_PATH = "../models/resnet50-bs=128"  # 存放模型权重的地方
    MODEL_CKPT_NAME = "e=8-s=384-acc=95.59.pth"

    set_seed(42) # 和train一致
    train_split, test_split, classes = split_data(DATA_PATH) # classes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    model = resnet50()
    hidden_dim = model.fc.weight.shape[1] # (1000, 2048)
    model.fc = torch.nn.Linear(hidden_dim, len(classes)) # 重新初始化一个Linear层
    model.device = device
    model.to(device)

    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, MODEL_CKPT_NAME),
                                     map_location=device))
    model.eval()

    while True:
        idx = int(input("enter a index:\n"))
        classify(test_split, idx, classes, model, preprocess)
        # ex: 12 Alakazam vs Kadabra
    print("Done!")