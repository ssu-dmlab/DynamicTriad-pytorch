import sys
from fire import Fire
from loguru import logger
from data import Dataset

def main(
	model='original',
	dir='datasets',
	dataset='academic_toy',
	epochs=10,
	lr=0.1,
	timestep=36,
	emb_dim=48,
	beta_triad=1.0,
	beta_smooth=1.0,
	batchsize=1000,
	mode='link_reconstruction',
):


	logger.debug("loading dataset")
	dataset = Dataset(dir + '/' + dataset, timestep)

	if model == 'original':
		from models.original.model import Model
		from models.original.train import Trainer
		from models.original.eval import Evaluator

		model = Model(
			len(dataset.vertices),
			timestep-1,
			emb_dim,
			params={
				'beta_triad': beta_triad,
				'beta_smooth': beta_smooth
			}
		)

		evaluator = Evaluator(mode)
		trainer = Trainer(model, dataset, evaluator)
	else:
		logger.error("no such model {}".format(model))
		return None

	trained_model = trainer.train(lr=lr, epochs=epochs, batchsize=batchsize)
	f1score = evaluator.evaluate(trained_model, dataset)

	logger.info(f1score)
	return f1score

if __name__ == "__main__":
	Fire(main)
