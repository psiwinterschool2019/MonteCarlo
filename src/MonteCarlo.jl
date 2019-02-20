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
