import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self, config):
        super(SimpleClassifier, self).__init__()
        self.config = config
        
        # Example improvement: Add bias and dropout, and switch to ReLU
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(config.hidden_dim, config.num_classes, bias=False),
        )
        
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, batch):
        model_output = {}

        # Squeeze if your embedding is shape (batch_size, 1, embedding_dim)
        # Otherwise, you can remove .squeeze(1) if not needed
        logits = self.classifier(batch['embedding'].squeeze(1))
        
        # Compute outputs
        model_output['id'] = batch['id']
        model_output['logits'] = logits
        model_output['probs'] = F.softmax(logits, dim=-1)
        model_output['preds'] = logits.argmax(dim=-1)
        model_output['labels'] = batch['label']

        labels = batch['label'].long()
        model_output['loss'] = self.loss_criterion(logits, labels)

        return model_output
