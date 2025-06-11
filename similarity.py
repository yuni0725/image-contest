import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# 1. Define the model
class PretrainedResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=128, pretrained=True):
        super().__init__()
        base_model = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(
            *list(base_model.children())[:-1]
        )  # remove avgpool + fc
        self.projector = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.backbone(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 512]
        x = self.projector(x)  # [B, output_dim]
        return x


# 2. Define the image transform (resize + ToTensor only, no normalization)
transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),  # scales to [0, 1]
    ]
)


# 3. Load image (PIL.Image)
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # add batch dimension → [1, 3, 512, 512]


# 4. Run model
if __name__ == "__main__":
    # Load and transform image
    image_tensor1 = load_image(
        "images/target_image.png"
    )  # <-- replace with your image path
    image_tensor2 = load_image(
        "generated_images/31203 권남윤_06-10-09-20_1749514800.png"
    )  # <-- replace with your image path

    # Instantiate model
    model = PretrainedResNetFeatureExtractor(output_dim=128)
    model.eval()  # switch to eval mode

    # Run inference
    with torch.no_grad():
        feature_vector1 = model(image_tensor1)
        feature_vector2 = model(image_tensor2)

    print(
        float(
            (feature_vector1 @ feature_vector2.T).item()
            / (np.linalg.norm(feature_vector1) * np.linalg.norm(feature_vector2))
        )
    )
