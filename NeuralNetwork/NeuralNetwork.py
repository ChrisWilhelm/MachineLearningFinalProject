import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

def read_data():
    sample_data = pd.read_csv('../dataset/Consolidated_CancerSEEK_Data.csv')
    sample_data = sample_data.drop(["Patient ID #", "Sample ID #", "Tumor type", "Race", "AJCC Stage"], axis=1)
    demographic_biomarker_data = sample_data.drop(["CancerSEEK Logistic Regression Score", "CancerSEEK Test Result"], axis=1)
    biomarker_data = demographic_biomarker_data.drop(["Age", "Sex"], axis=1)
    replicated_biomarker_data = biomarker_data[["Ω score", "CA-125 (U/ml)", "CA19-9 (U/ml)", "CEA (pg/ml)", "HGF (pg/ml)",
                                                "Myeloperoxidase (ng/ml)", "OPN (pg/ml)", "Prolactin (pg/ml)", "TIMP-1 (pg/ml)",
                                                "Ground Truth"]]
    return demographic_biomarker_data, biomarker_data, replicated_biomarker_data

demographic_biomarker_data, biomarker_data, replicated_biomarker_data = read_data()

class BestModel(torch.nn.Module):
	def __init__(self, input_dimensions):
		super().__init__()
		self.input_dimensions = input_dimensions
		# output dimensions is 1 (a binary classifier)

		self.linear1 = nn.Linear(input_dimensions, 127)
		self.linear2 = nn.Linear(127, 1)


	def forward(self, x):
		x = x.reshape(x.shape[0], 1, self.input_width, self.input_height)

		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))

		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x = F.relu(self.conv6(x))

		x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
		# print(x.shape)

		x = F.relu(self.linear1(x))
		x = self.linear2(x)

		return x

if __name__ == "__main__":
	MODE = arguments.get('mode')

	if MODE == "train":

		LOG_DIR = arguments.get('log_dir')
		MODEL_SAVE_DIR = arguments.get('model_save_dir')
		LEARNING_RATE = arguments.get('lr')
		BATCH_SIZE = arguments.get('bs')
		EPOCHS = arguments.get('epochs')
		DATE_PREFIX = datetime.datetime.now().strftime('%Y%m%d%H%M')
		if LEARNING_RATE is None: raise TypeError("Learning rate has to be provided for train mode")
		if BATCH_SIZE is None: raise TypeError("batch size has to be provided for train mode")
		if EPOCHS is None: raise TypeError("number of epochs has to be provided for train mode")
		TRAIN_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_images.npy"))
		TRAIN_LABELS = np.load(os.path.join(DATA_DIR, "fruit_labels.npy"))
		DEV_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_dev_images.npy"))
		DEV_LABELS = np.load(os.path.join(DATA_DIR, "fruit_dev_labels.npy"))

		N_IMAGES = TRAIN_IMAGES.shape[0]  # Number of images in the training corpus
		HEIGHT = TRAIN_IMAGES.shape[1]  # Height dimension of each image
		WIDTH = TRAIN_IMAGES.shape[2]  # width dimension of each image
		N_CLASSES = len(np.unique(TRAIN_LABELS))  # number of output classes
		N_DEV_IMAGES = DEV_IMAGES.shape[0]  # Number of images in the dev set

		flat_train_imgs = TRAIN_IMAGES.reshape(N_IMAGES,	 HEIGHT * WIDTH)
		flat_dev_imgs   = DEV_IMAGES  .reshape(N_DEV_IMAGES, HEIGHT * WIDTH)

		# Normalize each of the individual images to a mean of 0 and a variance of 1
		MEAN = np.average(flat_train_imgs)
		STD = np.std(flat_train_imgs)

		flat_train_imgs = (flat_train_imgs - MEAN) / STD
		flat_dev_imgs   = (flat_dev_imgs   - MEAN) / STD

		# do not touch the following 4 lines (these write logging model performance to an output file
		# stored in LOG_DIR with the prefix being the time the model was trained.)
		LOGFILE = open(os.path.join(LOG_DIR, f"bestmodel.log"),'w')
		log_fieldnames = ['step', 'train_loss', 'train_acc', 'dev_loss', 'dev_acc']
		logger = csv.DictWriter(LOGFILE, log_fieldnames)
		logger.writeheader()

		model = BestModel(HEIGHT, WIDTH, N_CLASSES)

		### TODO (OPTIONAL) : you can change the choice of optimizer here if you wish.
		optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

		for step in range(EPOCHS): # LR: 0.001, bs: 200, epochs: 4000
			i = np.random.choice(flat_train_imgs.shape[0], size=BATCH_SIZE, replace=False)
			x = torch.from_numpy(flat_train_imgs[i].astype(np.float32))
			y = torch.from_numpy(TRAIN_LABELS[i].astype(np.int))


			# Forward pass: Get logits for x
			logits = model(x)
			# Compute loss
			loss = F.cross_entropy(logits, y)
			# Zero gradients, perform a backward pass, and update the weights.
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# log model performance every 100 epochs
			if step % 100 == 0:
				train_acc, train_loss = approx_train_acc_and_loss(model, flat_train_imgs, TRAIN_LABELS)
				dev_acc, dev_loss = dev_acc_and_loss(model, flat_dev_imgs, DEV_LABELS)
				step_metrics = {
					'step': step,
					'train_loss': loss.item(),
					'train_acc': train_acc,
					'dev_loss': dev_loss,
					'dev_acc': dev_acc
				}

				print(f"On step {step}:\tTrain loss {train_loss}\t|\tDev acc is {dev_acc}")
				logger.writerow(step_metrics)
		LOGFILE.close()

		### TODO (OPTIONAL) You can remove the date prefix if you don't want to save every model you train
		### i.e. "{DATE_PREFIX}_bestmodel.pt" > "bestmodel.pt"
		model_savepath = os.path.join(MODEL_SAVE_DIR, "bestmodel.pt")

		print("Training completed, saving model at {model_savepath}")
		torch.save(model, model_savepath)


	elif MODE == "predict":
		PREDICTIONS_FILE = arguments.get('predictions_file')
		WEIGHTS_FILE = arguments.get('weights')
		if WEIGHTS_FILE is None : raise TypeError("for inference, model weights must be specified")
		if PREDICTIONS_FILE is None : raise TypeError("for inference, a predictions file must be specified for output.")
		TEST_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_test_images.npy"))

		N_IMAGES = TEST_IMAGES.shape[0]  # Number of images in the testing corpus
		HEIGHT = TEST_IMAGES.shape[1]  # Height dimension of each image
		WIDTH = TEST_IMAGES.shape[2]  # width dimension of each image

		model = torch.load(WEIGHTS_FILE)

		flat_test_imgs = TEST_IMAGES.reshape(N_IMAGES, HEIGHT * WIDTH)

		MEAN = np.average(flat_test_imgs)
		STD = np.std(flat_test_imgs)

		flat_test_imgs = (flat_test_imgs - MEAN) / STD

		predictions = []
		for test_case in flat_test_imgs:
			x = torch.from_numpy(test_case.astype(np.float32))
			x = x.view(1,-1)
			logits = model(x)
			pred = torch.max(logits, 1)[1]
			predictions.append(pred.item())
		print(f"Storing predictions in {PREDICTIONS_FILE}")
		predictions = np.array(predictions)
		np.savetxt(PREDICTIONS_FILE, predictions, fmt="%d")

	else: raise Exception("MODE not recognized")


def neural_net(dataset):
	np_dataset = dataset.to_numpy()
    num_col = np.shape(np_dataset)[1]
    X = np_dataset[:, :-1]
    y = np_dataset[:, -1]

	X_train = X[:-100]
	y_train = y[:-100]
	X_test  = X[-100:]  # put away - only to use at end of project
	y_test  = y[-100:]  # put away - only to use at end of project

	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
