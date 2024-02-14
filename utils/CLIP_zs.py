import torch.nn as nn
import torch
import clip
from tqdm.auto import tqdm
from utils.datasets import *

class CLIP_ZeroShot(nn.Module):
    def __init__(self, classes, clip_model, device):
        super().__init__()
        self.model = clip_model
        self.text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)

    def forward(self, image):
        batch_size = image.shape[0]

        text_inputs = self.text_inputs
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)

        value = torch.zeros(batch_size, dtype=torch.int64)
        index = torch.zeros(batch_size, dtype=torch.int64)

        for idx in range(batch_size):
            # Calculate features
            with torch.no_grad():
                image_features = self.model.encode_image(image[idx,:,:,:].unsqueeze(0))

            ## Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Pick the most similar label
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            value[idx], index[idx] = similarity[0].topk(1)

        return value, index
    
def test_clip(model, data_loader, device):
    model.eval()

    test_acc = 0

    with torch.inference_mode():
        for images, classes in tqdm(data_loader):
            images = images.to(device)
            _, pred = model(images)

            test_acc += (pred == classes).sum() / classes.shape[0]

        test_acc = test_acc / len(data_loader)
    return test_acc