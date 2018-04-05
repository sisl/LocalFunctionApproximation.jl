"""
This module implements various methods of locally approximating a function, such as multi-linear, simplex
and k-nearest-neighbor approximation. An example use case is for locally approximating value 
functions in value iteration
"""
__precompile__()
module LocalFunctionApproximation

using GridInterpolations
using NearestNeighbors
using Distances

export
	LocalFunctionApproximator,
	LocalGIFunctionApproximator,
	LocalNNFunctionApproximator,
	n_interpolants,
	get_all_interpolating_points,
	get_all_interpolating_values,
	get_interpolating_nbrs_idxs_wts,
	evaluate,
	batch_update


abstract type LocalFunctionApproximator end

"""
	n_interpolants(lfa::LocalFunctionApproximator)

Return the number of interpolants that the approximator is using
"""
function n_interpolants end

"""
	get_all_interpolating_points(lfa::LocalFunctionApproximator)

Return the vector of points (in a specific order) that are used to interpolate
"""
function get_all_interpolating_points end

"""
	get_all_interpolating_values(lfa::LocalFunctionApproximator)

Return the vector of all interpolating values (in the same order as the interpolating points)
"""
function get_all_interpolating_values end

"""
	get_interpolating_nbrs_idxs_wts(lfa::LocalFunctionApproximator, v::AbstractVector)

Return a tuple of (indices, weights) for the interpolants for a specific query v
"""
function get_interpolating_nbrs_idxs_wts end

"""
	evaluate(lfa::LocalFunctionApproximator, v::AbstractVector)

Return the value of the function at some query point v, based on the local function approximator
"""
function evaluate end

"""
	batch_update(lfa::LocalFunctionApproximator, vals::AbstractVector) 

Set the values of all interpolants to the input vector
"""
function batch_update end


include("local_gi_fa.jl")
include("local_nn_fa.jl")

end # module
