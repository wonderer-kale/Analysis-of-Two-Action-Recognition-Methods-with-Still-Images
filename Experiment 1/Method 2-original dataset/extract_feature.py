import torch
from torchvision import transforms
from torchvision.models import AlexNet_Weights

def extract_feature(images):
    outputs = []

    def hook_fn(m, i, o):
        outputs.append(o.clone().detach().squeeze(0))

    # Load the pretrained model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.IMAGENET1K_V1, verbose=False)
    model.eval()

    # sample execution (requires torchvision)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    handle = model.classifier[4].register_forward_hook(hook_fn)

    for image in images:
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            model(input_batch)
        
    handle.remove()

    outputs = torch.cat(outputs)
    return outputs.cpu().numpy() # n x 16384