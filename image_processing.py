import torch
import torchvision


class ImageInference:

    def __init__(
        self, model_name,
        num_classes=2, feature_extract=False, use_pretrained=True, pretrain_path=None
        ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained
        self.pretrain_path = pretrain_path
        self.model, self.input_size = self._initialize_model()
        if self.pretrain_path is not None:
            self._load_pretrained_model()
        self.model.eval()

    def _set_parameter_requires_grad(self, model):
        if self.feature_extract:
            for param in model.parameters():
                param.requires_grad = False

    def _initialize_model(self):
        model_ft = None
        input_size = 0
        if self.model_name == "resnet":
            """ Resnet18
            """
            model_ft = torchvision.models.resnet18(pretrained=self.use_pretrained)
            self._set_parameter_requires_grad(model_ft)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = torch.nn.Linear(num_ftrs, self.num_classes)
            input_size = 224
        elif self.model_name == "alexnet":
            """ Alexnet
            """
            model_ft = torchvision.models.alexnet(pretrained=self.use_pretrained)
            self._set_parameter_requires_grad(model_ft)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = torch.nn.Linear(num_ftrs, self.num_classes)
            input_size = 224
        elif self.model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = torchvision.models.vgg11_bn(pretrained=self.use_pretrained)
            self._set_parameter_requires_grad(model_ft)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = torch.nn.Linear(num_ftrs, self.num_classes)
            input_size = 224
        elif self.model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = torchvision.models.squeezenet1_0(pretrained=self.use_pretrained)
            self._set_parameter_requires_grad(model_ft)
            model_ft.classifier[1] = torch.nn.Conv2d(
                512, self.num_classes, kernel_size=(1,1), stride=(1,1)
                )
            model_ft.num_classes = self.num_classes
            input_size = 224
        elif self.model_name == "densenet":
            """ Densenet
            """
            model_ft = torchvision.models.densenet121(pretrained=self.use_pretrained)
            self._set_parameter_requires_grad(model_ft)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = torch.nn.Linear(num_ftrs, self.num_classes)
            input_size = 224
        elif self.model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = torchvision.models.inception_v3(pretrained=self.use_pretrained)
            self._set_parameter_requires_grad(model_ft)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = torch.nn.Linear(num_ftrs, self.num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = torch.nn.Linear(num_ftrs,self.num_classes)
            input_size = 299
        else:
            print("Invalid model name, exiting...")
            exit()
        return model_ft, input_size

    def _load_pretrained_model(self):
        self.model.load_state_dict(torch.load(self.pretrain_path))

    def preprocess(self, raw_image):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.input_size),
            torchvision.transforms.CenterCrop(self.input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transform(raw_image)
        return torch.unsqueeze(image, 0)

    def predict(self, image):
        with torch.no_grad():
            output = self.model(image)
        prediction = torch.nn.functional.softmax(output, dim=1)[0]
        return prediction

    def process_predict(self, image_raw):
        image = self.preprocess(image_raw)
        prediction = self.predict(image)
        return prediction