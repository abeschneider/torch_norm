require 'Norm'
require 'nn'

local precision = 1e-5
tester = torch.Tester()
mytest = {}

function mytest.TestNorm()
	local module = Norm(2)

	local input = torch.Tensor(10, 1):zero()
	local err = nn.Jacobian.testJacobian(module, input)
	print(err)
	tester:assertlt(err, precision, 'error on state ')
end

tester:add(mytest)
tester:run()
