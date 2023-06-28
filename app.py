from flask import Flask, jsonify, request
import base64
import os
import torch
from torch import nn
from torchvision import transforms
import torchvision

app = Flask(__name__)

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,  # how big is the square that's going over the image?
                      stride=1,  # default
                      padding=1),  # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)  # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units * 56 * 87,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion

def makePrediction(device,
                   model: torch.nn.Module,
                   image_path: str,
                   class_names: list[str] = None,
                   transform=None):
    """Makes a prediction on a target image and plots the image with its prediction."""

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    return class_names[target_image_pred_label.cpu()]

def cardRecognitionAlgorithm(folder_name, i):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_state_dict = ["Ace of Pentacles", "Ace of Cups", "Ace of Swords", "Ace of Wands", "Death", "Eight of Cups", "Eight of Pentacles", "Eight of Swords", "Eight of Wands", "Five of Cups", "Five of Pentacles", "Four of Cups", "Four of pentacles", "Five of Swords", "Five of Wands", "Four of Swords", "Four of Wands", "Justice", "Judgement", "King of Pentacles", "Knight of Pentacles", "Knight of Swords", "King of Swords", "King of Wands", "Knight of Cups", "Knight of Wands", "Nine of Cups", "Nine of Pentacles", "Nine of Swords", "Nine of Wands", "Page of Swords", "Page of Wands", "Page of Cups", "Page of Pentacles", "Queen of Cups", "Queen of Pentacles", "Queen of Swords", "Queen of Wands", "Seven of Cups", "Seven of Pentacles", "Six of Cups", "Six of Pentacles", "Seven of Swords", "Seven of Wands", "Six of Swords", "Six of Wands", "Strength", "Ten of Cups", "Ten of swords", "The Devil", "The Emperor", "The Fool", "The Hanged Man", "The Hermit", "The High Priestess", "The Lovers", "The Star", "The Sun", "The Tower", "Three of Cups", "Two of Cups", "Ten of Pentacles", "Ten of Wands", "The Chariot", "The Empress", "The Hierophant", "The Magican", "The Moon", "The World", "Three of Pentacles", "Three of Swords", "Three of Wands", "Two of Pentacles", "Two of Swords", "Two of Wands", "Wheel of Fortune ", "King of Cups", "Temperance"]
    model = NeuralNetworkModel(input_shape=3,
                               hidden_units=10,
                               output_shape=len(model_state_dict)).to(device)

    custom_image_tr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 348)),
        transforms.ToTensor()
    ])

    model.load_state_dict(torch.load("reducedSizeModel.pt"))
    model.eval()
    return makePrediction(device, model, folder_name, model_state_dict, custom_image_tr)


@app.route("/")
def hello():
    return "Hello World!"


@app.route('/process', methods=['POST'])
def process():
    json_data = request.get_json()
    user_id = json_data['id']
    number_of_photos = json_data['number_of_cards']

    folder_name = 'detectedCards/cards_from_user_' + str(user_id)
    os.makedirs(folder_name, exist_ok=True)

    for i in range(1, number_of_photos + 1):
        image_data = json_data['image_' + str(i)]
        image = base64.b64decode(image_data)
        file_path = folder_name + '/image_' + str(i) + '.jpg'
        with open(file_path, "wb") as f:
            f.write(image)

    response_list = {}

    for i in range(1, number_of_photos + 1):
        file_path = folder_name + '/image_' + str(i) + '.jpg'
        card_name = cardRecognitionAlgorithm(file_path, i)
        response_list[i-1] = card_name

    for file_name in os.listdir(folder_name):
        file_path = os.path.join(folder_name, file_name)
        os.remove(file_path)
    os.rmdir(folder_name)
    return jsonify(response_list)


if __name__ == '__main__':
    app.run()