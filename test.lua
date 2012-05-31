require 'nn'

require 'Norm'

local precision = 1e-5
tester = torch.Tester()
mytest = {}

function mytest.TestNorm()
	local module = Norm()

	local input = torch.Tensor(10, 1):zero()
	local err = nn.Jacobian.testJacobian(module, input)
	print(err)
	tester:assertlt(err, precision, 'error on state ')
end

function mytest.TestNormSquared()
	local module = Norm(true)

	local input = torch.Tensor(10, 1):zero()
	local err = nn.Jacobian.testJacobian(module, input)
	print(err)
	tester:assertlt(err, precision, 'error on state ')
end

function mytest.TestCDivTable()
	local input = torch.Tensor(10, 1):zero()	
	
	local seq = nn.Sequential()
	local split = nn.ConcatTable()
	seq:add(split)
	
	-- [1] vector
	split:add(nn.Identity())
	
	-- [2] normalization factor (need to replicate normalization so we can do a cdiv)
	local nseq = nn.Sequential()
	nseq:add(Norm())
	nseq:add(nn.Replicate(input:size(1)))
	split:add(nseq)
	
	-- vector / norm(vector)
	seq:add(nn.CDivTable())		
	
	local err = nn.Jacobian.testJacobian(seq, input)
	print(err)
	tester:assertlt(err, precision, 'error on state ')	
end


tester:add(mytest)
tester:run()
