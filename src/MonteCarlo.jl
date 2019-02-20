using Random
using Statistics

module MonteCarlo

function magnetization(c::BitArray{2})::Float64  #This is the magnetization function
    Lx,Ly=size(c)
    N=Lx*Ly
    Nup=count(c)
    
    (2Nup-N)/N
end

end

########################################################################

function make_bins(v::Array{T,1}, bin_length::Int64) where T<:Number
    
    nbins = length(v)÷bin_length
    output = zeros(nbins)
    
    for i in 1:nbins
        output[i] = mean(v[((i-1)*bin_length +1) : i*bin_length])
    end
        
    return output
end

########################################################################
