import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ResNet(nn.Module):
    def __init__(self, num_classes, output='both'):
        '''
        Constructs and returns ResNet model.
        '''
        super().__init__()
        self._net = resnet50(pretrained=False)
        assert output in ['predictions', 'confidence', 'both']
        in_features = self._net.fc.in_features
        self._net = torch.nn.Sequential(*list(self._net.children())[:-1])
        self.classifier = nn.Linear(in_features, num_classes)
        self.confidence = nn.Linear(in_features, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.output = output

    def forward(self, inputs):
        '''
        Applies model to inputs. Returns the task_scores logits and the confidence logits
        '''
        out = self._net(inputs)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        confidence_score = self.confidence(out)
        task_scores = self.classifier(out)
        task_scores = F.log_softmax(task_scores, dim=1)
        if self.output == 'predictions':
            return task_scores
        elif self.output == 'confidence':
            return confidence_score
        return task_scores, confidence_score
