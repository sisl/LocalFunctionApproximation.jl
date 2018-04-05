using LocalFunctionApproximation
using Base.Test
using NearestNeighbors
using StaticArrays

# Only testing NNFA as the GIFA is just a wrapper
# for the already tested GridInterpolations package
points = [SVector(0.,0.), SVector(0.,1.), SVector(1.,1.), SVector(1.,0.)]
nntree = KDTree(points)
vals = [1., 1., -1., -1]
k = 2
r = 0.5*sqrt(2)

knnfa = LocalNNFunctionApproximator(nntree, points, k)
batch_update(knnfa, vals)

@test n_interpolants(knnfa) == 4

for i = 1:10
    point = [rand()/2, 0.5]
    @test LocalFunctionApproximation.evaluate(knnfa, point) == 1.0
end

for i = 1:10
    point = [1.0 - rand()/2., 0.5]
    @test LocalFunctionApproximation.evaluate(knnfa, point) == -1.0
end

rnnfa = LocalNNFunctionApproximator(nntree, points, r)
batch_update(rnnfa,vals)
for i = 1:10
    point = [rand()/2, 0.5]
    @test LocalFunctionApproximation.evaluate(rnnfa, point) == 1.0
end

for i = 1:10
    point = [1.0 - rand()/2., 0.5]
    @test LocalFunctionApproximation.evaluate(rnnfa, point) == -1.0
end