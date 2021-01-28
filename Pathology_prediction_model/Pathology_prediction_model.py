import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from copy import deepcopy
import torch
from torch.nn import Linear, MSELoss, CrossEntropyLoss, LogSoftmax, NLLLoss, functional as F
from torch.optim import SGD, Adam, RMSprop
from sklearn.model_selection import KFold


df = pd.read_excel("Supplementary file 2-2. Clinical data.xlsx", 'Sheet1')
df_train = df.iloc[:, 2:7]

df_train = pd.get_dummies(df_train)
label_columns = ['Histology_' + i for i in ['Normal', 'Inflammation', 'LGIN', 'HGIN', 'SM1', 'MM', 'SM2 or deeper']]
labels = df_train.loc[:, label_columns]
labels_gt = np.argmax(np.array(labels), 1)
data = df_train.drop(label_columns, axis=1)
data = np.array(data)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=10000, shuffle=True)

df_IPCLsNet = pd.read_excel(p + "\\Clinical_results.xlsx", 'Sheet1')
df_IPCLsNet = df_IPCLsNet.iloc[:, 2:6]
df_IPCLsNet = np.array(df_IPCLsNet)


# df_IPCLsNet = torch.from_numpy(df_IPCLsNet).to(device).float()
# df_IPCLsNet = df_IPCLsNet / torch.sum(df_IPCLsNet, 1, keepdim=True)


def random_shuffle(data, df_IPCLsNet, label):
	randnum = np.random.randint(0, 1200)
	np.random.seed(randnum)
	np.random.shuffle(data)
	np.random.seed(randnum)
	np.random.shuffle(df_IPCLsNet)
	np.random.seed(randnum)
	np.random.shuffle(label)
	return data, df_IPCLsNet, label


def cagegrary_normalize(x_train, y_train):
	x_train = torch.from_numpy(x_train).to(device).float()
	x_train = x_train / torch.sum(x_train, 1, keepdim=True)
	y_train = torch.from_numpy(y_train).to(device).long()
	return x_train, y_train


# test the model
def eval_model(model, x_test, y_test):
	model.eval()
	reg_accuracy = torch.mean((torch.argmax(model(x_test), 1) == y_test) * 1.0)
	print('MSE: {}'.format(reg_accuracy))
	return torch.mean(reg_accuracy)


# define the model
class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = Linear(4, 64)
		self.fc2 = Linear(64, 28)
		self.fc3 = Linear(28, 28)
		self.fc4 = Linear(28, 7)
	
	def forward(self, x):
		x = F.leaky_relu(self.fc1(x))
		x = F.leaky_relu(self.fc2(x))
		x = F.leaky_relu(self.fc3(x))
		x = self.fc4(x)
		return x


def data_augumentation(x_train, y_train):
	for i in torch.unique(y_train):
		id = i == y_train
		A = torch.rand((torch.sum(id), torch.sum(id)), device=device)
		A = A / torch.sum(A, 1, keepdim=True)
		x_train[id] = torch.mm(A, x_train[id])
	return x_train, y_train


# define the number of epochs and the data set size
nb_epochs = 1000
step = 25
data_size = 1000


def train_model(foldi, x_train, x_test, y_train, y_test, rec):
	model = Net()
	model = model.to(device)
	
	print('# generator parameters:', sum(param.numel() for param in model.parameters()))
	
	# define the loss function
	critereon = CrossEntropyLoss()
	# define the optimizer
	optimizer = Adam(model.parameters(), lr=0.0003, weight_decay=0.01)
	accuracy_best = 0
	for i, epoch in enumerate(range(nb_epochs)):
		# break
		model.train()
		epoch_loss = 0
		for ix in range(x_train.shape[0]):
			N = 64
			idx = np.random.randint(0, x_train.shape[0], N * 2)
			x = x_train[idx[:N], :]
			y = y_train[idx[:N]]
			x, y = data_augumentation(x, y)
			y_pred = model(x)
			loss = critereon(y_pred, y)
			epoch_loss = loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print("Epoch: {} Loss: {}".format(epoch, epoch_loss))
		
		if epoch % step == 0:
			train_loss = critereon(model(x_train), y_train)
			rec['train_loss'].append(train_loss.detach().cpu().numpy().item())
			
			test_loss = critereon(model(x_test), y_test)
			rec['test_loss'].append(test_loss.detach().cpu().numpy().item())
			eval_model(model, x_train, y_train)
			accuracy = eval_model(model, x_test, y_test)
			
			train_accuracy = eval_model(model, x_train, y_train)
			rec['train_accuracy'].append(train_accuracy.detach().cpu().numpy().item())
			
			test_accuracy = eval_model(model, x_test, y_test)
			rec['test_accuracy'].append(test_accuracy.detach().cpu().numpy().item())
			
			if accuracy_best < accuracy:
				best_val_model = deepcopy(model.state_dict())
				torch.save(model, 'trained_models/NetFCN_Fold{}.pt'.format(foldi))
				accuracy_best = accuracy
	model.load_state_dict(best_val_model)
	return model


data, df_IPCLsNet, labels = random_shuffle(data, df_IPCLsNet, labels_gt)
data, _ = cagegrary_normalize(data, labels)
df_IPCLsNet, labels = cagegrary_normalize(df_IPCLsNet, labels)

kf = KFold(n_splits=5)
fold_accuracy = []
IPCLsNet_accuracy = []
rec = {'train_step': [], 'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': []}

for i, (train, test) in enumerate(kf.split(X=data, y=labels)):
	print("fold: %d/10" % i)
	x_train, x_test = data[train, :], data[test, :]
	y_train, y_test = labels[train], labels[test]
	
	IPCLsNet_test = df_IPCLsNet[test, :]
	
	# x_train, x_test, y_train, y_test = torch_from_numpy(x_train, x_test, y_train, y_test)
	model = train_model(i, x_train, x_test, y_train, y_test, rec)
	rec['train_step'].extend(np.arange(nb_epochs // step))
	fold_accuracy.append(eval_model(model, x_test, y_test).item())
	IPCLsNet_accuracy.append(eval_model(model, IPCLsNet_test, y_test).item())

np.save('rec.npy', rec)
np.save('fold_accuracy.npy', fold_accuracy)
np.save('IPCLsNet_accuracy.npy', IPCLsNet_accuracy)
print('mean accuracy: {}, std: {}'.format(np.mean(fold_accuracy), np.std(fold_accuracy)))
print('mean accuracy: {}, std: {}'.format(np.mean(IPCLsNet_accuracy), np.std(IPCLsNet_accuracy)))

rec_ = np.load('rec.npy', allow_pickle=True).item()
fold_accuracy_ = np.load('fold_accuracy.npy', allow_pickle=True)
IPCLsNet_accuracy_ = np.load('IPCLsNet_accuracy.npy', allow_pickle=True)
df = pd.DataFrame.from_dict(rec_)

df = pd.read_excel('rec.xlsx', sheet_name='Sheet1')
sns.lineplot(x="Training epoch", y="Training loss", data=df)
plt.savefig('Training loss.svg')
plt.show()

df = pd.read_excel('rec.xlsx', sheet_name='Sheet2')
sns.lineplot(x="Training epoch", y="Accuracy", hue='Data', data=df)
plt.legend(loc='lower right')
plt.savefig('Accuracy.svg')
plt.show()
