package.path = package.path..";/n/rush_lab/wonlee/hidden-gsn/?.lua"
require 'rnn'
require 'hdf5'

require 'gsn'
require 's2sa.data'
require 's2sa.models'
require 's2sa.model_utils'
require 's2sa.memory'

cmd = torch.CmdLine()
-- Training hyperparameters
cmd:option('-epochs', 30, 'epochs of GSN training')
cmd:option('-max_grad_norm', 5, 'maximum gradient norm clipping')
cmd:option('-learning_rate', 0.7, 'learning rate')
-- GSN model hyperparameters
cmd:option('-criterion', 'mse', 'criterion for GSN loss')
cmd:option('-noise', 5, 'variance for GSN')
cmd:option('-prob', 0.5, 'geometric probability')
cmd:option('-enc_dim', 500, 'dimension of hidden state of encoder')
cmd:option('-hid_dim', 300, 'dimension of GSN reconstruction hidden layer')
-- File data
cmd:option('-data_file','convert/ptb-s2s-train.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-val_data_file','convert/ptb-s2s-val.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-gpu', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-savefile', 'enc_samples.hdf5','filename to save samples to')
cmd:option('-modelfile', 'checkpoint/ptb-s2s_final.t7', 'filename to load model from')
cmd:option('-prealloc', 1, 'if preallocate memory')

opt = cmd:parse(arg)

function forwardConnect(enc, dec)
	for i = 1, #enc.lstmLayers do
		local seqlen = #enc.lstmLayers[i].outputs
		dec.lstmLayers[i].userPrevOutput = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevOutput, enc.lstmLayers[i].outputs[seqlen])
		dec.lstmLayers[i].userPrevCell = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevCell, enc.lstmLayers[i].cells[seqlen])
	end
end

function mergeTables(t1, t2)
	for k, v in pairs(t2) do t1[k] = v end
	return t1
end

function append_table(dst, src)
  for i = 1, #src do
      table.insert(dst, src[i])
  end
end

function zero_table(t)
  for i = 1, #t do
      if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
      	 if i == 1 then
	    cutorch.setDevice(opt.gpuid)
	 else
	    cutorch.setDevice(opt.gpuid2)
	 end
      end
      t[i]:zero()
  end
end

function reset_state(state, batch_l, t)
    if t == nil then
        local u = {}
	for i = 1, #state do
	    state[i]:zero()
	    table.insert(u, state[i][{{1, batch_l}}])
	end
	return u
    else
	local u = {[t] = {}}
	for i = 1, #state do
	    state[i]:zero()
	    table.insert(u[t], state[i][{{1, batch_l}}])
	end
	return u
    end
end

-- Turn on memory preallocation
preallocateMemory(opt.prealloc)

function train(data, valid_data)
    print("Training GSN...")
    -- Preliminary setup
	context_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)

	-- clone encoder/decoder up to max source/target length
	--decoder_clones = clone_many_times(decoder, opt.max_sent_l_targ)
 	encoder_clones = clone_many_times(encoder, opt.max_sent_l_src)
	for i = 1, opt.max_sent_l_src do
	    --encoder_clones[i] = encoder_clones[i]:cuda()
	    if encoder_clones[i].apply then
	       encoder_clones[i]:apply(function(m) m:setReuse() end)
	       if opt.prealloc == 1 then encoder_clones[i]:apply(function(m) m:setPrealloc() end) end
	    end
	end
	--for i = 1, opt.max_sent_l_targ do
	--    if decoder_clones[i].apply then
	--       decoder_clones[i]:apply(function(m) m:setReuse() end)
	--       if opt.prealloc == 1 then decoder_clones[i]:apply(function(m) m:setPrealloc() end) end
	--    end
	--end

	local h_init = torch.zeros(opt.max_batch_l, opt.rnn_size)
	local attn_init = torch.zeros(opt.max_batch_l, opt.max_sent_l)
	if opt.gpuid >= 0 then
           h_init = h_init:cuda()
	   attn_init = attn_init:cuda()
	   cutorch.setDevice(opt.gpuid)
	   context_proto = context_proto:cuda()
	   --encoder_grad_proto = encoder_grad_proto:cuda()
	end

  	-- these are initial states of encoder/decoder for fwd/bwd steps
    	init_fwd_enc = {}
      	init_bwd_enc = {}
        init_fwd_dec = {}
	init_bwd_dec = {}
	for L = 1, opt.num_layers do
     	    table.insert(init_fwd_enc, h_init:clone())
            table.insert(init_fwd_enc, h_init:clone())
	    table.insert(init_bwd_enc, h_init:clone())
	    table.insert(init_bwd_enc, h_init:clone())
	end
	table.insert(init_bwd_dec, h_init:clone())
	for L = 1, opt.num_layers do
            table.insert(init_fwd_dec, h_init:clone())
	    table.insert(init_fwd_dec, h_init:clone())
	    table.insert(init_bwd_dec, h_init:clone())
	    table.insert(init_bwd_dec, h_init:clone())
	end

    for t = 1, opt.epochs do
	-- SGD loop
	local loss = 0
	local total = 0
	local batch_order = torch.randperm(data.length)
	for i = 1, data:size() do
	    --zero_table(grad_params, 'zero')
            local d = data[batch_order[i]]
	    local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
	    local batch_l, target_l, source_l = d[5], d[6], d[7]
	    if opt.gpuid >= 0 then
	       cutorch.setDevice(opt.gpuid)
	    end
	    local rnn_state_enc = reset_state(init_fwd_enc, batch_l, 0)
	    local context = context_proto[{{1, batch_l}, {1, source_l}}]
	    -- forward prop encoder
      	    for t = 1, source_l do
            	encoder_clones[t]:training()
	        local encoder_input = {source[t]}
		if data.num_source_features > 0 then
		   append_table(encoder_input, source_features[t])
		end
		append_table(encoder_input, rnn_state_enc[t-1])
		local out = encoder_clones[t]:forward(encoder_input)
		rnn_state_enc[t] = out
		context[{{},t}]:copy(out[#out])
	    end
	    -- encoder hidden state top layer
	    local final_hid = rnn_state_enc[source_l][opt.num_layers*2]
	    print(final_hid:size())

	    -- GSN f/b training
	    gsn:forward(final_hid)
	    loss = loss + opt.enc_dim * batch_l * gsn:backward(final_hid)
	    total = total + opt.enc_dim * batch_l  	    
	    print("Epoch: ", t, "Training err: ", loss/total)	    
	end
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
		gsn:forward(encoder.lstmLayers[#encoder.lstmLayers].outputs[sentlen-1])
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
	local model = torch.load(opt.modelfile)
	encoder = model[1][1]
	encoder = encoder:cuda()
	decoder = model[1][2]
	generator = model[1][3]
	opt = mergeTables(model[2], opt)
	gsn = GSN(opt.noise, nil, opt.criterion, opt.prob, opt.max_grad_norm, opt.learning_rate, opt.enc_dim, opt.hid_dim, opt.gpu)
	print("Models loaded!")
	
	-- Load data
	local train_data = data.new(opt, opt.data_file)
	local valid_data = data.new(opt, opt.val_data_file)
	print("Data loaded!")
	
	-- Train
	train(train_data, valid_data)--, encoder, gsn)
	-- Check/save results
	torch.save(opt.modelfile .. '_gsn.t7', gsn)
end

main()
