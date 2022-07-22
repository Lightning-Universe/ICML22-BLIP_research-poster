# This is code is inspired from the official BLIP implementation by Salesforce
#  Credits: https://github.com/salesforce/BLIP
import os
from re import S
import requests
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
from models.blip_vqa import blip_vqa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_demo_image(raw_image, image_size, device):
    # img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    w, h = raw_image.size

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


class CaptionModel:
    def __init__(self) -> None:
        self.image_size = 384
        model_url = (
            "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
        )

        model = blip_decoder(pretrained=model_url, image_size=self.image_size, vit="base")
        model.eval()
        self.model = model.to(device)

    def predict(self, image):
        image = load_demo_image(image_size=self.image_size, device=device)
        with torch.no_grad():
            # beam search
            caption = self.model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
            # nucleus sampling
            # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
            return caption[0]


class VQAModel:
    def __init__(self) -> None:

        self.image_size = 480
        model_url = (
            "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
        )

        model = blip_vqa(pretrained=model_url, image_size=self.image_size, vit="base")
        model.eval()
        self.model = model.to(device)

    def predict(self, image, question):
        image = load_demo_image(image_size=self.image_size, device=device)

        with torch.no_grad():
            answer = self.model(image, question, train=False, inference="generate")
            return answer[0]

class Model:
    def __init__(self) -> None:
        os.chdir("BLIP")
        self.vqa = VQAModel()
        self.caption  = CaptionModel()

    def predict(self, image, task:str, question: str):
        if task=="Visual Question Answering":
            return self.vqa.predict(image=image, question=question)
        else:
            return self.caption.predict(image=image)
