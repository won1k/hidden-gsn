package.path = package.path..";/n/rush_lab/wonlee/hidden-gsn/?.lua"
require 'nn'
require 'hdf5'
require 'nngraph'
require 'distributions'
require 'utils'

local GSN = torch.class("GSN")

-- Currently use squared-error loss (nn.MSECriterion())
-- noise = "gaussian", ...
-- model: should be nn module (i.e. MLP) with :forward/:backward pass
function GSN:__init(noise, model, criterion, prob, max_grad_norm, learning_rate, ndim, nhid, gpu)
	self.noise = noise
	self.model = model or utils.make_model(ndim, nhid)
	self.criterion = criterion or nn.MSECriterion()
	self.prob = prob
	self.params, self.gradParams = self.model:getParameters()
	self.gpu = gpu or 1

	self.samples = torch.Tensor
	self.currLoss = 0
	self.prevLoss = 1e9
	self.maxGradNorm = max_grad_norm or 5
	self.learningRate = learning_rate or 0.7
end

-- :forward expects nbatch x ndim tensor [states]
function GSN:forward(states)
	states = states:double()
	local nbatch = states:size(1)
	local ndim = states:size(2)
	local k = torch.geometric(self.prob)
	local currState = states:clone()
	for i = 1, k do
		currState = distributions.mvn.rnd(currState, currState, torch.eye(ndim))
		currState = self.model:forward(currState):clone()
	end
	self.samples = currState
	if self.gpu > 0 then
		return currState:cuda()
	else
		return currState
	end
end

-- :backward expects nbatch x ndim tensor [states], samples (or uses previous)
function GSN:backward(states, samples)
	local currSamples
	if samples then
		currSamples = samples
	else
		currSamples = self.samples
	end
	local pred = self.model:forward(currSamples)
	local loss = self.criterion:forward(pred, states)
	local gradOutput = self.criterion:backward(pred, states)
	self.model:backward(currSamples, gradOutput)
	local gradNorm = self.gradParams:norm()
	if gradNorm > self.maxGradNorm then
		self.gradParams:mul(self.maxGradNorm / gradNorm)
	end
	self.params:add(self.gradParams:mul(-self.learningRate))
	return loss
end