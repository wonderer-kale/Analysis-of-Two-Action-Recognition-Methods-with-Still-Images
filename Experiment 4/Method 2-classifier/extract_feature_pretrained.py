import torch
import torch.nn as nn
from torchvision import transforms, models
import copy


def extract_feature(images):
    outputs = []

    def hook_fn(m, i, o):
        outputs.append(o.clone().detach().squeeze(0))

    # Initailize model
    net = models.alexnet()
    net.classifier[6] = nn.Linear(in_features=4096, out_features=12, bias=True)
    net.train()
    agent_model = copy.deepcopy(net)
    object_model = copy.deepcopy(net)
    union_model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Conv2d(32, 64, kernel_size=5),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Conv2d(64, 32, kernel_size=5),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(5408, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 12),
        nn.Sigmoid()
    )
    intersection_model = models.alexnet(weights='DEFAULT')
    intersection_model.eval()
    #intersection_model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=AlexNet_Weights.IMAGENET1K_V1, verbose=False)

    # Load the pretrained model
    agent_model.load_state_dict(torch.load('./weight/hs_model_weights_improved_dataset.pth'))
    agent_model.eval()
    object_model.load_state_dict(torch.load('./weight/os_model_weights_improved_dataset.pth'))
    object_model.eval()
    union_model.load_state_dict(torch.load('./weight/ps_model_weights_improved_dataset.pth'))
    union_model.eval()

    # sample execution (requires torchvision)
    default_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class SquarePad:
        def __call__(self, image):
            w, h = image.size
            max_wh = max(w, h)
            hp = (max_wh - h) // 2
            wp = (max_wh - w) // 2
            padding = (wp, hp, wp, hp)
            return transforms.functional.pad(image, padding, fill=0, padding_mode='constant')

    pretrained_preprocess = transforms.Compose([
        SquarePad(),
        transforms.Resize(128),
        transforms.ToTensor(),
    ])
    
    agent_handle = agent_model.classifier[4].register_forward_hook(hook_fn)
    intersection_model_handle = intersection_model.classifier[4].register_forward_hook(hook_fn)
    object_handle = object_model.classifier[4].register_forward_hook(hook_fn)
    union_handle = union_model[14].register_forward_hook(hook_fn)
    
    for i, image in enumerate(images):
        pretrained_input_tensor = pretrained_preprocess(image)
        default_input_tensor = default_preprocess(image)
        pretrained_input_batch = pretrained_input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        default_input_batch = default_input_tensor.unsqueeze(0)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            pretrained_input_batch = pretrained_input_batch.to('cuda')
            default_input_batch = default_input_batch.to('cuda')
            agent_model.to('cuda')
            intersection_model.to('cuda')
            object_model.to('cuda')
            union_model.to('cuda')

        with torch.no_grad():
            if i == 0:
                agent_model(pretrained_input_batch)
            elif i == 1:
                intersection_model(default_input_batch)
            elif i == 2:
                object_model(pretrained_input_batch)
            else:
                union_model(pretrained_input_batch)

    agent_handle.remove()
    intersection_model_handle.remove()
    object_handle.remove()
    union_handle.remove()

    outputs = torch.cat(outputs)
    return outputs.cpu().numpy() # n x 16384