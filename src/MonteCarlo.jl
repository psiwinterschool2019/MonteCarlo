using Random
using Statistics

module MonteCarlo

spin(b::Bool) = Float64(2 * b - 1)

neighbours(nx::Int, ny::Int, Lx::Int, Ly::Int) = ((mod1(nx+1, Lx), ny), (mod1(nx-1, Lx), ny), 
    (nx, mod1(ny+1, Ly)), (nx, mod1(ny-1, Ly)))

function local_energy(c::AbstractMatrix{Bool}, nx::Int, ny::Int, h::Float64, Jx::Float64, Jy::Float64)
    """
  returns: the local energy (wrt to nearest neighbours) of the spin configuration
    """
    Lx,Ly = size(c)
    nb1, nb2, nb3, nb4 = neighbours(nx, ny, Lx, Ly)
    b = c[nx,ny]
    s = spin(b)
    sn_x = spin(c[nb1[1], nb1[2]]) + spin(c[nb2[1], nb2[2]])
    sn_y = spin(c[nb3[1], nb3[2]]) + spin(c[nb4[1], nb4[2]])
    E = -(h + Jx * sn_x + Jy * sn_y)*s
    return E
end

function energy(c::AbstractMatrix{Bool}, h::Float64, Jx::Float64, Jy::Float64  )
    """
  returns: the total energy of the spin configuration
    """
    Lx,Ly= size(c)
    E=0.0
    m = spin.(c) 
    for k in 1:Lx, j in 1:Ly
        E-= h* m[k,j]
        E-= Jx* m[k,j]* m[mod1(k+1,Lx),j]
        E-= Jy* m[k,j]* m[k,mod1(j+1,Ly)]
    end
    return E
end

function update!(c::AbstractMatrix{Bool}, β::Float64, h::Float64, Jx::Float64, Jy::Float64)
    """
  Do one sweep (random one single-spin update):
  returns: true if the metropolis move is accepted or false otherwise
    """
    Lx,Ly = size(c)
    nx = rand(1:Lx)
    ny = rand(1:Ly)
    deltaE = - 2*local_energy(c,nx,ny,h,Jx,Jy) #variation
    
    w = exp(-β * deltaE)
    accepted = false
    if ( rand() < w)
        c[nx,ny] = !c[nx,ny] #Flipping spin at nx,ny
        accepted = true
    end
    return accepted
    
end

function magnetization(c::AbstractMatrix{Bool})  
    """
    returns: mean magnetization of the spin configuration
    """
    Lx,Ly=size(c)
    N=Lx*Ly
    mean_magnetization = sum(spin.(c))/N
   
    return mean_magnetization
end

function make_bins(v::Array{T,1}, bin_length::Int64) where T<:Number
    
    nbins = length(v)÷bin_length
    output = zeros(nbins)
    
    for i in 1:nbins
        output[i] = mean(v[((i-1)*bin_length +1) : i*bin_length])
    end
        
    return output
end

end
