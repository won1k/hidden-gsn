package.path = package.path .. '/n/rush_lab/wonlee/hidden-gsn/?.lua
require 'hdf5'
require 'nn'

--require 's2sa.scorers.bleu'
--require 'train-encoder'

cmd = torch.CmdLine()

cmd:option('-gpu', 1, 'use gpu')
cmd:option('-auto', 1, 'autoencoder')

local data = torch.class("data")
function data:__init(data_file)
    local f = hdf5.open(data_file, 'r')
    self.lengths = f:read('sent_lens'):all()
    self.length = self.lengths:size(1)
    self.sents = {}
    for i = 1, self.length do
    	local l = self.lengths[i]
    	self.sents[l] = f:read(tostring(len)):all():double()
    end
    f:close()
end

local function get_ngrams(s, n, count)
    local ngrams = {}
    count = count or 0
    for i = 1, #s do
    	for j = i, math.min(i+n-1, #s) do
	    local ngram = table.concat(s, ' ', i, j)
	    local l = j-i+1 -- keep track of ngram length
	    if count == 0 then
	        table.insert(ngrams, ngram)
	    else
		if ngrams[ngram] == nil then
		    ngrams[ngram] = {1, l}
		else
		    ngrams[ngram][1] = ngrams[ngram][1] + 1
		end
	    end
	end
    end
    return ngrams
end

local function get_ngram_prec(cand, ref, n)
    -- n = number of ngrams to consider
    local results = {}
    for i = 1, n do
    	results[i] = {0, 0} -- total, correct
    end
    local cand_ngrams = get_ngrams(cand, n, 1)
    local ref_ngrams = get_ngrams(ref, n, 1)
    for ngram, d in pairs(cand_ngrams) do
    	local count = d[1]
	local l = d[2]
	results[l][1] = results[l][1] + count
	local actual
	if ref_ngrams[ngram] == nil then
	    actual = 0
	else
	    actual = ref_ngrams[ngram][1]
	end
	results[l][2] = results[l][2] + math.min(actual, count)
    end
    return results
end

-- BLEU score for sentence-grouped file
function get_bleu(cand, ref)
    assert(type(cand) == 'string', 'need candidate file name')
    assert(type(ref) == 'string', 'need ref file name')
    cand = data.new(cand)
    ref = data.new(ref)

    -- Compute ngram precision
    local ngram_prec = {}
    for n = 1, 4 do
    	ngram_prec[n] = {0, 0}
    end
    local total_pred_length = 0
    local total_gold_length = 0
    for i = 1, cand.length do
    	local l = cand.lengths[i]
    	local pred_sents = cand[l] -- nsent x sent_len tensor
	local gold_sents = ref[l]
	local nsent = pred_sents:size(1)
	for j = 1, nsent do
	    local pred_sent = torch.sum(torch.clamp(pred_sents[j],0,1))
    	    total_pred_length = torch_pred_length + pred_sent
	    total_gold_length = total_gold_length + l
	    local prec = get_ngram_prec(pred_sents[j], gold_sents[j], 4)
	    for n = 1, 4 do
	    	ngram_prec[n][1] += prec[n][1]
	    	ngram_prec[n][2] += prec[n][2]
	    end
	end
    end

    -- Compute BLEU
    local bleu = 0
    local brev_penalty = math.exp(1 - math.exp(1, total_gold_length/total_pred_length)
    for n = 1, 4 do
    	bleu = bleu + math.log(ngram_prec[n][2]/ngram_prec[n][1])
    end
    return brev_penalty * math.exp(bleu/4)
end