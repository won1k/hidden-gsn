require 'nn';
require 'hdf5'
require 'nngraph'

local GSN = torch.class("GSN")

function GSN:__init(noise, model, steps)
	self.noise = noise
	self.model = model
	self.steps = steps
end