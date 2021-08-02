from data import Dataset
from models.original.model import Model
from models.original.train import Trainer

dataset = Dataset('../datasets/academic_toy', 18)
model = Model(len(dataset.vertices), 18, 48)
trainer = Trainer(model, dataset)
trainer.train()
