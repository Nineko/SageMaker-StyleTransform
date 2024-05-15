import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import cv2

from Utils.common import image_loader,save_image_to_s3,ContentLoss,StyleLoss,Normalization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers,
                               style_layers):
    cnn = copy.deepcopy(cnn)

    # 规范化模块
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # 只是为了拥有可迭代的访问权限或列出内容/系统损失
    content_losses = []
    style_losses = []

    # 假设cnn是一个`nn.Sequential`，
    # 所以我们创建一个新的`nn.Sequential`来放入应该按顺序激活的模块
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # 加入内容损失:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # 加入风格损失:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # 现在我们在最后的内容和风格损失之后剪掉了图层
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]



    return model, style_losses, content_losses



def get_input_optimizer(input_img):
    # input是我們決定需要grafient的parameter
    # optimizer = optim.LBFGS([input_img.requires_grad_()])
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img,input_img,
                       content_layers_default,style_layers_default,
                       num_steps=150,style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img,content_layers_default,style_layers_default)
    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)
    optimizer = get_input_optimizer(input_img)
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                 input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            return style_score + content_score

        optimizer.step(closure)

    # 最后的修正......
    with torch.no_grad():
         input_img.data.clamp_(0, 1)
    return input_img

def train(args):
    style_img = image_loader(args.style_image)
    content_img = image_loader(args.content_image)
    save_path = args.output_data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vgg19 = models.vgg19(pretrained=False)
    vgg19.load_state_dict(torch.load("VGG19PreTrain/vgg19.pth"))
    cnn = vgg19.features.eval().to(device)

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    input_img = content_img.clone()

    model_output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img,content_layers_default,style_layers_default,
                                num_steps = args.num_steps)

    save_image_to_s3(model_output,'sagemaker-us-east-1-851725197925','Result/result.jpg')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument('--num-steps', type=int, default=150)

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--content-image', type=str, default=os.environ['SM_CHANNEL_CONTENT'])
    parser.add_argument('--style-image', type=str, default=os.environ['SM_CHANNEL_STYLE'])

    train(parser.parse_args())



