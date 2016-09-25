package.path = package.path..";/n/rush_lab/wonlee/hidden-gsn/?.lua"
require 'rnn'
require 'hdf5'
require 'gsn'

cmd = torch.CmdLine()
-- Training hyperparamters
cmd:option('-epochs', 30, 'epochs of GSN training')
cmd:option('-max_grad_norm', 5, 'maximum gradient norm clipping')
cmd:option('-learning_rate', 0.7, 'learning rate')
-- GSN model hyperparameters
cmd:option('-criterion', 'mse', 'criterion for GSN loss')
cmd:option('-noise', 'gaussian', 'noise for GSN')
cmd:option('-prob', 0.5, 'geometric probability')
cmd:option('-enc_dim', 650, 'dimension of hidden state of encoder')
cmd:option('-hid_dim', 300, 'dimension of GSN reconstruction hidden layer')
-- File data
cmd:option('-data_file','convert/ptb.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-val_data_file','convert/ptb_test.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-gpu', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-savefile', 'enc_samples.hdf5','filename to save samples to')
cmd:option('-encfile', 'checkpoint/ptb_', 'filename to load encoder from')

opt = cmd:parse(arg)

function forwardConnect(enc, dec)
	for i = 1, #enc.lstmLayers do
		local seqlen = #enc.lstmLayers[i].outputs
		dec.lstmLayers[i].userPrevOutput = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevOutput, enc.lstmLayers[i].outputs[seqlen])
		dec.lstmLayers[i].userPrevCell = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevCell, enc.lstmLayers[i].cells[seqlen])
	end
end

-- Construct the data set.
local data = torch.class("data")
function data:__init(opt, data_file)
	local f = hdf5.open(data_file, 'r')
	self.input = {}
	self.output = {}
	self.lengths = f:read('sent_lens'):all()
	self.max_len = f:read('max_len'):all()[1]
	self.nfeatures = f:read('nfeatures'):all():long()[1]
	self.nclasses = f:read('nfeatures'):all():long()[1]
	self.length = self.lengths:size(1)
	for i = 1, self.length do
		local len = self.lengths[i]
		self.input[len] = f:read(tostring(len)):all():double()
		self.output[len] = self.input[len]
		if opt.gpu > 0 then
			self.input[len] = self.input[len]:cuda()
			self.output[len] = self.output[len]:cuda()
		end
	end
	f:close()
end

function data:size()
	return self.length
end

function data.__index(self, idx)
	local input, target
	if type(idx) == "string" then
		return data[idx]
	else
		input = self.input[idx]:transpose(1,2)
		output = self.output[idx]:transpose(1,2)
	end
	return {input, output}
end

function train(data, valid_data, encoder, gsn)
	print("Training GSN...")
	for t = 1, opt.epochs do
		for i = 1, data:size() do
			local sentlen = data.lengths[i]
			print("Sentence length: ", sentlen)
			local d = data[sentlen]
			local input, output = d[1], d[2]
	        local nsent = input:size(2)
	        -- Encoder forward
	        local encoderOutput = encoder:forward(input[{{1, sentlen - 1}}])
	        -- GSN sampling (start with just highest-layer hidden state; then do rest if works)
	        --for i = 1, #encoder.lstmLayers do
			--	encoder.lstmLayers[i].outputs[sentlen-1]
			--	encoder.lstmLayers[i].cells[sentlen-1]
			--end
	        gsn:forward(encoder.lstmLayers[#encoder.lstmLayers].outputs[sentlen-1]):clone()
	        gsn:backward(encoder.lstmLayers[#encoder.lstmLayers].outputs[sentlen-1])
	        -- Forget for next
			encoder:forget()
		end
		local score = eval(valid_data, encoder, gsn)
		print("Epoch: ", t, "Score: ", score)
	end
end

function eval(data, encoder, gsn)
	local loss = 0
	local total = 0
	for i = 1, data:size() do
		local sentlen = data.lengths[i]
		local d = data[sentlen]
		local input, output = d[1], d[2]
		local nbatch = input:size(1)
		local encoderOutput = encoder:forward(input[{{1, sentlen - 1}}])
		encoder.lstmLayers[#encoder.lstmLayers].outputs[sentlen-1] = 
	        gsn:forward(encoder.lstmLayers[#encoder.lstmLayers].outputs[sentlen-1]):clone()
		loss = loss + opt.enc_dim * nbatch * gsn:backward(encoder.lstmLayers[#encoder.lstmLayers].outputs[sentlen-1])
		total = total + opt.enc_dim * nbatch
		encoder:forget()
	end
	return loss/total
end

function main()
	-- Check if GPU
	if opt.gpu >= 0 then
		print("Running on GPU...")
		require 'cutorch'
		require 'cunn'
	end
	-- Load models
	local encoder = torch.load(opt.encfile .. 'encoder.t7')
	encoder:forget()
	local gsn = GSN(opt.noise, nil, opt.criterion, opt.prob, opt.max_grad_norm, opt.learning_rate, opt.enc_dim, opt.hid_dim, opt.gpu)
	print("Models loaded!")
	-- Load data
	local train_data = data.new(opt, opt.data_file)
	local valid_data = data.new(opt, opt.val_data_file)
	print("Data loaded!")
	-- Train
	train(train_data, valid_data, encoder, gsn)
	-- Check/save results
	torch.save(opt.encfile .. 'gsn.t7', gsn)
end

main()
