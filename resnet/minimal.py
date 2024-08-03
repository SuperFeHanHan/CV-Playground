# minimal code to use resnet

from torchvision.models import resnet50, ResNet50_Weights
# from torchvision.io import read_image
from PIL import Image

weights = ResNet50_Weights.IMAGENET1K_V2
preprocess = weights.transforms()
categories = weights.meta["categories"] # 之前模型训练的分类名称
model = resnet50(weights=weights)
model.eval()

# img = read_image("computer_keyboard.jpeg") # Tensor (3, 163, 310)
img = Image.open("computer_keyboard.jpeg") # 直接用PIL也可以
img_transformed = preprocess(img)          # Tensor (3, 224, 224)
img_transformed = img_transformed.unsqueeze(0) # (N, C, H, W)
output = model(img_transformed)  # (N, #classes)
score = output.squeeze(0).softmax(dim=-1)
pred = output.argmax(dim=-1).item()
print(f"Prediction: {categories[pred]}, Score = {score[pred]:.2f}")