require 'rnn'
require 'hdf5'
require 'nngraph'

cmd = torch.CmdLine()

cmd:option('-test_data_file','convert/ptb_test.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-gpu', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-outputfile', 'output/ptb-encoder.hdf5','filename to save predictions')
cmd:option('-loadfile', 'checkpoint/ptb-ae-final.t7', 'filename to load encoder/decoder from, if any')

opt = cmd:parse(arg)

END = 3

-- Construct the data set.
local data = torch.class("data")
function data:__init(opt, data_file)
   local f = hdf5.open(data_file, 'r')
   self.input = {}
   self.output = {}
   self.lengths = f:read('sent_lens'):all()
   self.max_len = f:read('max_len'):all()[1]
   self.nfeatures = f:read('nfeatures'):all():long()[1]
   if opt.auto == 1 then
     self.nclasses = f:read('nfeatures'):all():long()[1]
   else
     self.nclasses = f:read('nclasses'):all():long()[1]
   end
   self.length = self.lengths:size(1)
   self.dwin = opt.dwin
   for i = 1, self.length do
     local len = self.lengths[i]
     self.input[len] = f:read(tostring(len)):all():double()
     if opt.auto == 1 then
       self.output[len] = self.input[len]
     else
       self.output[len] = f:read(tostring(len) .. "_target"):all():double()
     end
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

function concatTable(t1,t2)
    for k,v in pairs(t2) do t1[k] = v end
    return t1
end

-- Connect functions for encoder-decoder
function forwardConnect(enc, dec)
   for i = 1, #enc.lstmLayers do
      local seqlen = #enc.lstmLayers[i].outputs
      dec.lstmLayers[i].userPrevOutput = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevOutput, enc.lstmLayers[i].outputs[seqlen])
      dec.lstmLayers[i].userPrevCell = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevCell, enc.lstmLayers[i].cells[seqlen])
      --dec.lstmLayers[i].userPrevOutput = enc.lstmLayers[i].outputs[seqlen]:clone()
      --dec.lstmLayers[i].userPrevCell = enc.lstmLayers[i].cells[seqlen]:clone()
   end
end

function predict(data, encoder, decoder)
   -- Testing
   encoder:evaluate()
   decoder:evaluate()

   -- Test file
   local f = hdf5.open(opt.outputfile, "w")
   f:write('sent_lens', data.lengths)   

   -- Setup metrics
   local nll = 0
   local total = 0
   local accuracy = 0

   -- Prediction
   for i = 1, data:size() do
      local sentlen = data.lengths[i]
      local d = data[sentlen]
      local input, output = d[1], d[2]
      if opt.ptb > 0 then
        sentlen = sentlen - 1
      end
      local nsent = input:size(2)
      local revInput
      if opt.rev > 0 then
        revInput = {}
        for t = 1, sentlen do
          table.insert(revInput, input[{{sentlen - t + 1},{}}])
        end
        input = nn.JoinTable(1):forward(revInput)
        if opt.gpu > 0 then
          input = input:cuda()
        end
      end
      output = nn.SplitTable(1):forward(output)

      -- Encoder forward prop
      local encoderOutput = encoder:forward(input[{{1, sentlen}}]) -- sentlen table of batch_size x rnn_size
      -- Decoder forward prop
      forwardConnect(encoder, decoder)
      local decoderInput = { input[{{sentlen + 1}}] }
      decoder:remember()
      local decoderOutput = { decoder:forward(decoderInput[1])[1]:clone() }
      nll = nll + criterion:forward(decoderOutput[1], output[1]) * nsent
      total = total + nsent
      for t = 2, sentlen + 1 do
        local _, nextInput = decoderOutput[t-1]:max(2)
        table.insert(decoderInput, nextInput:reshape(1,nsent):clone())
        table.insert(decoderOutput, decoder:forward(decoderInput[t])[1]:clone())
	nll = criterion:forward(decoderOutput[t], output[t]) * nsent
	total = total + nsent
      end

      -- Save predictions/accuracy (0 if past END)
      local predictions = torch.zeros(nsent, sentlen)
      for  t = 1, #decoderOutput do
      	   local _, pred = decoderOutput[t]:max(2)
	   pred = pred:reshape(nsent)
	   for i = 1, nsent do
	       	if predictions[i][t-1] == 0 or predictions[i][t-1] == END then
	       	    predictions[i][t] = 0
	       	else
		    predictions[i][t] = pred[i]
		end
		if predictions[i][t] == output[t][i] then
		    accuracy = accuracy + 1
		end
	   end		    
      end
      f:write(tostring(sentlen), predictions)

      encoder:forget()
      decoder:forget()
   end

   -- Summary/close
   f:close()
   local valid = math.exp(nll / total)
   print("Test error", valid)
   print("Test accuracy", accuracy / total)
   return valid, accuracy/total
end

function main()
    -- parse input params
   opt = cmd:parse(arg)

   if opt.gpu > 0 then
      print('using CUDA on GPU ' .. opt.gpu .. '...')
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(opt.gpu)
   end
   
   -- Load models
   if opt.loadfile:len() > 0 then
      print("Loading old models...")
      local model = torch.load(opt.loadfile)
      encoder = model[1]
      decoder = model[2]
      opt = tableConcat(model[3], opt)
   end

   if opt.gpu > 0 then
      encoder:cuda()
      decoder:cuda()
      criterion:cuda()
   end

   -- Create the data loader class.
   local test_data = data.new(opt, opt.test_data_file)

   -- Prediction
   predict(test_data, encoder, decoder)
end

main()
