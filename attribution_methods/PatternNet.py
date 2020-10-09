import torch
from tqdm import tqdm
from attribution_methods.pytorch_patternNet import PatternNetSignalEstimator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SignalEstimator:
    def __init__(self, model):
        self.model = model
        self.signal_estimator = PatternNetSignalEstimator(self.model)

    def train_explain(self, train_dataloader):
        with torch.no_grad():
            for batch, _ in tqdm(train_dataloader):
                self.signal_estimator.update_E(batch.to(device))

        self.signal_estimator.get_patterns()

    def __call__(self, batch,label):
        signal = self.signal_estimator.get_signal(batch.to(device),label)
        return signal


class Explainer:
    def __init__(self, model, signal_estimator):
        self.model = model
        self.signal_estimator=signal_estimator
        #self.signal_estimator = SignalEstimator(model)
        #self.signal_estimator.train_explain(train_dataloader)

    def get_attribution_map(self, batch,label=None):
        signal = self.signal_estimator(batch,label)
        return signal

