require 'nn'
require 'Norm'

local Normalizer, parent = torch.class('Normalizer', 'nn.Module')

function Normalizer:__init(size)
	parent.__init(self)
	
	self.seq = nn.Sequential()
	
	local split = nn.ConcatTable()
	self.seq:add(split)
	
	-- [1] vector
	split:add(nn.Identity())
	
	-- [2] normalization factor (need to replicate normalization so we can do a cdiv)
	local nseq = nn.Sequential()
	nseq:add(Norm())
	
	-- don't give size yet (we can calculate it later based on our input vector)
	self.replicate = nn.Replicate(0)
	nseq:add(self.replicate)
	split:add(nseq)
	
	-- vector / norm(vector)
	self.seq:add(nn.CDivTable())
end

function Normalizer:updateOutput(input)
	-- set number of features for replicate equal to size of input vector
	self.replicate.nfeatures = input:size(1)
	
	self.output = self.seq:forward(input)
	return self.output
end

function Normalizer:updateGradInput(input, gradOutput)
	self.gradInput = self.seq:updateGradInput(input, gradOutput)
	return self.gradInput
end
