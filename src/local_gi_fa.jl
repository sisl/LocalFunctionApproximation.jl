mutable struct LocalGIFunctionApproximator{G<:AbstractGrid} <: LocalFunctionApproximator
	grid::G
	gvalues::Vector{Float64}
end
LocalGIFunctionApproximator(grid::G) where {G<:AbstractGrid} = LocalGIFunctionApproximator(grid,zeros(length(grid)))


################ INTERFACE FUNCTIONS ################
function n_interpolating_points(gifa::LocalGIFunctionApproximator)
    return length(gifa.grid)
end

function get_all_interpolating_points(gifa::LocalGIFunctionApproximator)
	return vertices(gifa.grid)
end

function get_all_interpolating_values(gifa::LocalGIFunctionApproximator)
	return gifa.gvalues
end

function get_interpolating_nbrs_idxs_wts(gifa::LocalGIFunctionApproximator, v::AbstractVector{Float64})
    return interpolants(gifa.grid, v)
end

function compute_value(gifa::LocalGIFunctionApproximator, v::AbstractVector{Float64})
    return interpolate(gifa.grid, gifa.gvalues, v)
end

function compute_value(gifa::LocalGIFunctionApproximator, v_list::Vector{Vector{Float64}})
    @assert length(v_list) > 0
    vals = [compute_value(gifa, pt) for pt in v_list]
    return vals
end

function set_all_interpolating_values(gifa::LocalGIFunctionApproximator, gvalues::AbstractVector{Float64})
    gifa.gvalues = copy(gvalues)
end