require 'nn'

function make_model(ndim, nhid)
	local hid_dim = nhid or 300
	local model = nn.Sequential()
	model:add(nn.Linear(ndim, hid_dim))
	model:add(nn.Tanh())
	model:add(nn.Linear(hid_dim, ndim))
	return model
end