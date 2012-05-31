require 'nn'
local Norm, parent = torch.class('Norm', 'nn.Module')

function Norm:__init()
	parent.__init(self)
	
	self.gradInput = torch.Tensor()
	self.output = torch.Tensor(1)
	self.norm = p
end

function Norm:updateOutput(input)
	-- \sqrt{\sum_k x_k^{2}}
	self.output[1] = input:norm(2)
	return self.output
end

function Norm:updateGradInput(input, gradOutput)
	-- derivative of \sqrt(\sum_k x_k^{2})
	-- = \frac{x_i}{\sqrt{\sum x_k^{2}}}	
	local tmp = input:clone():pow(2)
	local div = torch.sqrt(torch.sum(tmp))
    self.gradInput:resizeAs(input):copy(input):div(div)
    self.gradInput:mul(gradOutput[1]);
	
    return self.gradInput
end

