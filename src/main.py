import sys
from fire import Fire
from loguru import logger
from data import Dataset

def main(
	model='original',
	dir='datasets',
	dataset='academic',
	device='cpu',
	epochs=10,
	lr=0.1,
	time_length=36,
	time_step=4,
	time_stride=2,
	emb_dim=48,
	beta_triad=1.0,
	beta_smooth=1.0,
	batchsize=10000,
	batdup=1,
	batchtqdm=True,
	mode='link_reconstruction',
):

	logger.debug("loading dataset")
	dataset = Dataset(dir + '/' + dataset, time_length, time_step, time_stride)

	if model == 'original':
		from models.original.model import Model
		from models.original.train import Trainer
		from models.original.eval import Evaluator

		model = Model(
			len(dataset.vertices),
			len(dataset)-1,
			emb_dim,
			params={
				'beta_triad': beta_triad,
				'beta_smooth': beta_smooth
			},
			device=device
		)

		evaluator = Evaluator(mode)
		trainer = Trainer(model, dataset, device, evaluator)
	else:
		logger.error("no such model {}".format(model))
		return None

	trained_model = trainer.train(lr=lr, epochs=epochs, batchsize=batchsize, batdup=batdup, batchtqdm=batchtqdm)
	f1score = evaluator.evaluate(trained_model, dataset)

	logger.info(f1score)
	return f1score

if __name__ == "__main__":
	Fire(main)
