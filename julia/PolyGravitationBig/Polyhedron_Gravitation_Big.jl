module Polyhedron_Gravitation_Big

include(joinpath(@__DIR__, "PolyhedronBase_Big.jl"))
include(joinpath(@__DIR__, "PolyhedronPotential_Big.jl"))
include(joinpath(@__DIR__, "PolyhedronAcceleration_Big.jl"))
include(joinpath(@__DIR__, "PolyhedronTensor_Big.jl"))

using .PolyhedronBaseBig
using .PolyhedronPotentialBig
using .PolyhedronAccelerationBig
using .PolyhedronTensorBig

export PolyhedronGravityBig,
       build_polyhedron_big,
       potential_big,
       acceleration_big,
       gravity_tensor_big

end # module
