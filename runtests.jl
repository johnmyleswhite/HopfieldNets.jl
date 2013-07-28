tests = ["test/HopfieldNets.jl", "test/storkey.jl"]

for path in tests
	include(path)
end
