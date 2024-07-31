from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import ResNet50_Weights
from torchvision.io import read_image
import matplotlib.pyplot as plt

def count_parameters(model):
    cnt = 0
    for k,v in model.named_parameters():
        cnt += v.numel()
    return cnt

def show_tensor(tensor):
    # tensor: (3, H, W)
    tensor = tensor.permute(1, 2, 0)
    plt.imshow(tensor.cpu().numpy())
    plt.show()

if __name__ == '__main__':
    # 研究模型参数量
    # https://pytorch.org/vision/main/models/resnet.html
    '''
    model18 = resnet18()
    model34 = resnet34()
    model50 = resnet50()
    model101 = resnet101()
    model152 = resnet152()

    print("模型参数")
    print(f"resnet18 {count_parameters(model18) / 10 ** 6:.2f} M param")
    print(f"resnet34 {count_parameters(model34) / 10 ** 6:.2f} M param")
    print(f"resnet50 {count_parameters(model50) / 10 ** 6:.2f} M param")
    print(f"resnet101 {count_parameters(model101) / 10 ** 6:.2f} M param")
    print(f"resnet152 {count_parameters(model152) / 10 ** 6:.2f} M param")
    '''

    # QuickDemo
    # https://pytorch.org/vision/stable/models.html
    weights = ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    categories = weights.meta["categories"] # 之前模型训练的分类名称
    model = resnet50(weights=weights)
    model.eval()

    img = read_image("computer_keyboard.jpeg") # Tensor (3, 163, 310)
    img_transformed = preprocess(img)          # Tensor (3, 224, 224)
    img_transformed = img_transformed.unsqueeze(0) # (N, C, H, W)
    output = model(img_transformed)  # (N, #classes)
    score = output.squeeze(0).softmax(dim=-1)
    pred = output.argmax(dim=-1).item()
    print(f"Prediction: {categories[pred]}, Score = {score[pred]:.2f}")

    # 研究一下模型结构
