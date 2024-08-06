import logging
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from transformers import SwinForImageClassification
# from .efficientnet_pytorch.model import EfficientNet
from timm import create_model
import timm

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
        )

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet121.classifier(out)
        return out


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.args = args
        logging.info(f"=> creating model '{args.visual_extractor}'")
        if args.visual_extractor == 'densenet':
            self.model = DenseNet121(args.num_labels)
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

            if args.pretrain_cnn_file and args.pretrain_cnn_file != "":
                logging.info(f'Load pretrained CNN model from: {args.pretrain_cnn_file}')
                checkpoint = torch.load(args.pretrain_cnn_file, map_location='cuda:{}'.format(args.gpu))
                self.model.load_state_dict(checkpoint['state_dict'])
        
        elif 'swin' in args.visual_extractor:
            # self.visual_extractor = args.visual_extractor
            # self.model = create_model('swin_base_patch4_window7_224', pretrained=True)
            # self.model.head = nn.Identity()  # Remove the classification head
            # # Pooling layer for Transformer outputs
            # self.avg_fnt = nn.AdaptiveAvgPool2d((1, 1))
            # self.reduce_dim = nn.Linear(in_features=1024, out_features=768)
            # self.classifier = nn.Linear(768, args.num_labels)
            
            pretrained = False if args.pretrain_cnn_file and len(args.pretrain_cnn_file) > 0 else True
            if not pretrained:
                model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=18)
                # 加载自己预训练的权重
                model.load_state_dict(torch.load(args.pretrain_cnn_file).state_dict())
                logging.info(f'Loaded pretrain cnn file: {args.pretrain_cnn_file}')
            else:
                model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)
                logging.info('Loaded torchvision pretrained file')

            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.avg_fnt = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(model.num_features, args.num_labels)

                        
        elif 'resnet' in self.args.visual_extractor.visual_extractor:
            self.visual_extractor = args.visual_extractor
            # 是否加载Pytorch ZOO的预训练权重
            pretrained = False if args.pretrain_cnn_file and len(args.pretrain_cnn_file) > 0 else True
            if not pretrained:
                model = models.resnet101(num_classes=18, pretrained=pretrained)
                model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                # 加载自己预训练的权重
                model.load_state_dict(torch.load(args.pretrain_cnn_file).state_dict())
                logging.info(f'Loaded pretrain cnn file: {args.pretrain_cnn_file}')
                model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            else:
                model = models.resnet101(pretrained=pretrained)
                logging.info('Loaded torchvision pretrained file')

            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            self.classifier = nn.Linear(2048, args.num_labels)
        else:
            raise NotImplementedError

    def forward(self, images):
        if self.args.visual_extractor == 'densenet':
            patch_feats = self.model.densenet121.features(images)
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))

            x = F.relu(patch_feats, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            labels = self.model.densenet121.classifier(x)
            
        elif 'swin' in self.args.visual_extractor:
            patch_feats = self.model(images)
            # print(patch_feats.shape)        # torch.Size([64, 1024, 7, 7])
            # avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
            patch_feats = patch_feats.permute(0, 3, 1, 2)
            avg_feats = self.avg_fnt(patch_feats).squeeze()

            # print(avg_feats.shape)        # torch.Size([64, 1024])
            labels = self.classifier(avg_feats)
            
        # elif self.args.visual_extractor == 'swin':
        #     patch_feats = self.model(images)
        #     print(patch_feats.shape)        # torch.Size([64, 7, 7, 1024])
            
        #     if patch_feats.dim() == 4:
        #         patch_feats = patch_feats.permute(0, 3, 1, 2)
        #         avg_feats = self.avg_fnt(patch_feats)  # Shape: [64, 1024, 1, 1]    torch.Size([64, 7, 1, 1])
        #         print(f"Shape after avg_fnt: {avg_feats.shape}")    # torch.Size([64, 7])
                
        #         # Flatten the tensor to remove spatial dimensions
        #         avg_feats = avg_feats.flatten(start_dim=1)  # Shape: [64, 1024] 
        #         print(f"Shape after flattening: {avg_feats.shape}") 

        #     else:
        #         raise RuntimeError(f"Unexpected shape of patch_feats: {patch_feats.shape}")

        #     # Ensure avg_feats has the correct size for the classifier
        #     print(f"avg_feats shape: {avg_feats.shape}")  
        #     avg_feats = self.reduce_dim(avg_feats)
        #     print("---------------here----------------")
        #     print(f"avg_feats shape: {avg_feats.shape}")  
            
        #     labels = self.classifier(avg_feats)
        #     print("here")
            
            
        elif self.args.visual_extractor == 'efficientnet':
            patch_feats = self.model.extract_features(images)
            # Pooling and final linear layer
            avg_feats = self.model._avg_pooling(patch_feats)
            x = avg_feats.flatten(start_dim=1)
            x = self.model._dropout(x)
            labels = self.model._fc(x)
            
        elif 'resnet' in self.visual_extractor:
            patch_feats = self.model(images)
            # print(patch_feats.shape)        # torch.Size([64, 2048, 7, 7])
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
            # print(avg_feats.shape)        # torch.Size([64, 2048])
            labels = self.classifier(avg_feats)
        else:
            raise NotImplementedError

        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        # print(patch_feats.shape, avg_feats.shape, labels.shape)
        return patch_feats, avg_feats, labels

torch.autograd.set_detect_anomaly(True)