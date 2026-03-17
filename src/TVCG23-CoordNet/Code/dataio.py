from utils import *
import numpy as np
import torch
import skimage 
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import data,img_as_float
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os

class ScalarDataSet():
	def __init__(self,args):
		self.dataset = args.dataset
		self.batch_size = args.batch_size
		self.application = args.application
		self.interval = args.interval
		self.scale = args.scale
		self.factor = args.factor
		allowed_datasets = {'H2', 'He', 'H+', 'GT', 'PD'}
		if self.dataset not in allowed_datasets:
			raise ValueError('Unsupported dataset: {}. Only {} are supported.'.format(self.dataset, sorted(allowed_datasets)))

		self.dim = [600,248,248]
		self.total_samples = 100
		self.local_npy = os.path.join(args.data_root, 'target_{}.npy'.format(self.dataset))
		if not os.path.exists(self.local_npy):
			raise FileNotFoundError('Local data not found: {}'.format(self.local_npy))

		os.makedirs(args.result_path+args.dataset, exist_ok=True)
		os.makedirs(args.model_path+args.dataset, exist_ok=True)

		if self.application == 'extrapolation':
			self.training_samples = self.total_samples*8//10
			self.samples = range(1,self.training_samples+1)
		elif self.application == 'temporal':
			self.samples = [i for i in range(1,self.total_samples+1,self.interval+1)]
			self.total_samples = self.samples[-1]
		elif self.application == 'spatial':
			self.samples = range(1,self.total_samples+1)
		elif self.application == 'super-spatial':
			if self.dataset == 'PD':
				self.samples = range(51,self.total_samples+1)
				self.total_samples = 50
			else:
				self.samples = range(1,self.total_samples+1)


	def GetCoords(self):
		if self.application == 'extrapolation':
			self.coords = get_mgrid([self.total_samples,self.dim[0],self.dim[1],self.dim[2]],dim=4,s=1,t=0)
			self.coords = self.coords[0:self.training_samples*self.dim[0]*self.dim[1]*self.dim[2]:,]
		elif self.application == "temporal":
			self.coords = get_mgrid([self.total_samples,self.dim[0],self.dim[1],self.dim[2]],dim=4,s=1,t=self.interval)
		elif self.application == 'spatial':
			self.Subsample()
			self.coords = get_mgrid([self.total_samples,self.dim[0],self.dim[1],self.dim[2]],dim=4,s=self.scale,t=0)
		elif self.application == 'super-spatial':
			self.coords = get_mgrid([self.total_samples,self.dim[0]*self.scale,self.dim[1]*self.scale,self.dim[2]*self.scale],dim=4,s=self.scale,t=0)

	def ReadData(self):
		self.GetCoords()
		self.data = []
		samples_per_timestep = self.dim[0] * self.dim[1] * self.dim[2]
		data_memmap = np.load(self.local_npy, mmap_mode='r')
		for i in self.samples:
			print(i)
			start = (i - 1) * samples_per_timestep
			end = start + samples_per_timestep
			d = np.asarray(data_memmap[start:end], dtype=np.float32).reshape(-1)
			d_min = np.min(d)
			d_max = np.max(d)
			if d_max > d_min:
				d = 2*(d-d_min)/(d_max-d_min)-1
			if self.application == 'spatial':
				self.data += list(d[self.coords_indices])
			else:
				self.data += list(d)
		self.data = np.asarray(self.data)


	def Subsample(self):
		self.coords_indices = []
		for z in range(0,self.dim[2],self.scale):
			for y in range(0,self.dim[1],self.scale):
				for x in range(0,self.dim[0],self.scale):
					index = (((z) * self.dim[1] + y) * self.dim[0] + x)
					self.coords_indices.append(index)
		self.coords_indices = np.asarray(self.coords_indices)


	def GetTrainingData(self):
		indices = []
		if self.application == 'spatial':
			samples = (self.dim[0]*self.dim[1]*self.dim[2])//(self.scale*self.scale*self.scale)
		elif self.application == 'super-spatial':
			samples = (self.dim[0]*self.dim[1]*self.dim[2])
		else:
			samples = self.dim[0]*self.dim[1]*self.dim[2]


		if self.application == 'extrapolation':
			for i in range(0,self.training_samples):
				index = np.random.choice(np.arange(i*samples,(i+1)*samples), self.factor*self.batch_size, replace=False)
				indices += list(index)
		elif self.application == 'temporal':
			for i in range(0,len(self.samples)):
				index = np.random.choice(np.arange(i*samples,(i+1)*samples), self.factor*self.batch_size, replace=False)
				indices += list(index)
		elif self.application == 'super-spatial':
			for i in range(0,len(self.samples)):
				index = np.random.choice(np.arange(i*samples,(i+1)*samples), self.factor*self.batch_size, replace=False)
				indices += list(index)


		if self.application in ['temporal','completion','super-spatial','extrapolation']:
			training_data_input = torch.FloatTensor(self.coords[indices])
			training_data_output = torch.FloatTensor(self.data[indices])
		elif self.application == 'spatial':
			if self.factor*self.batch_size >= samples:
				training_data_input = torch.FloatTensor(self.coords)
				training_data_output = torch.FloatTensor(self.data)
			else:
				for i in range(0,len(self.samples)):
					index = np.random.randint(low=i*samples,high=(i+1)*samples,size=self.factor*self.batch_size)
					indices += list(index)
				training_data_input = torch.FloatTensor(self.coords[indices])
				training_data_output = torch.FloatTensor(self.data[indices])
		data = torch.utils.data.TensorDataset(training_data_input,training_data_output)
		train_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True)
		return train_loader

	def GetTestingData(self):
		return get_mgrid([self.total_samples,self.dim[0],self.dim[1],self.dim[2]],dim=4)
		


class AODataSet():
	def __init__(self,args):
		raise ValueError('AODataSet has been removed. Only H2/He/H+/GT/PD local ionization datasets are supported.')

class ViewSynthesis():
	def __init__(self,args):
		raise ValueError('ViewSynthesis dataset loading has been removed. Only H2/He/H+/GT/PD local ionization datasets are supported.')
