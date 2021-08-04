from loguru import logger
from data import Dataset
from models.original.model import Model
from models.original.train import Trainer
from models.original.eval import Evaluator

dataset = Dataset('datasets/academic_toy', 36)
model = Model(len(dataset.vertices), 35, 48)
trainer = Trainer(model, dataset)
trained_model = trainer.train()
evaluator = Evaluator('link_reconstruction')
logger.info(evaluator.evaluate(model, dataset))
