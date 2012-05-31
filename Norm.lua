require 'nn'
local Norm, parent = torch.class('Norm', 'nn.Module')

function Norm:__init(squared_norm)
	parent.__init(self)
	
	self.gradInput = torch.Tensor()
	self.output = torch.Tensor(1)
	self.squared_norm = squared_norm or false
end

function Norm:updateOutput(input)
	if not self.squared_norm then
		-- \sqrt{\sum_k x_k^{2}}
		self.output[1] = input:norm(2)
	else
		-- \sum_k x_k^{2}
		self.output[1] = torch.sum(input:clone():pow(2))
		return self.output
	end
	
	return self.output	
end

function Norm:updateGradInput(input, gradOutput)
	if not self.squared_norm then
		-- derivative of \sqrt(\sum_k x_k^{2})
		-- = \frac{x_i}{\sqrt{\sum x_k^{2}}}	
		local tmp = input:clone():pow(2)
		local div = torch.sqrt(torch.sum(tmp))
	    self.gradInput:resizeAs(input):copy(input):div(div)
	    self.gradInput:mul(gradOutput[1]);
	else
		-- derivative of \sum_k x_k^{2}
		-- = 2*x_i
		self.gradInput:resizeAs(input):copy(input)
		self.gradInput:mul(2)
	end
	
    return self.gradInput
end

