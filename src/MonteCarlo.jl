module MonteCarlo

using Random
using Statistics
using StatsBase
using Plots

spin(b::Bool) = Float64(2 * b - 1)

neighbours(nx::Int, ny::Int, Lx::Int, Ly::Int) = ((mod1(nx+1, Lx), ny), (mod1(nx-1, Lx), ny), 
    (nx, mod1(ny+1, Ly)), (nx, mod1(ny-1, Ly)))

function local_energy(c::AbstractMatrix{Bool}, nx::Int, ny::Int, Jx::Float64, Jy::Float64, h::Float64)
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

function energy(c::AbstractMatrix{Bool}, Jx::Float64, Jy::Float64, h::Float64)
    """
  returns: the total energy of the spin configuration
    """
    Lx,Ly= size(c)
    E=0.0
     
    for k in 1:Lx, j in 1:Ly
        E-= h* spin(c[k,j])
        E-= Jx* spin(c[k,j])* spin(c[mod1(k+1,Lx),j])
        E-= Jy* spin(c[k,j])* spin(c[k,mod1(j+1,Ly)])
    end
    return E
end


function magnetization(c::AbstractMatrix{Bool})  
    """
    returns: mean magnetization of the spin configuration
    """
    Lx,Ly=size(c)
    N=Lx*Ly
    mean_magnetization = 2*count(c)/N - 1.0
   
    return mean_magnetization
end

function meas_mag(c::AbstractMatrix{Bool}, Jx::Float64, Jy::Float64, h::Float64)
    """
    returns: mean magnetization of the spin configuration
    """
    return magnetization(c)
end

function update!(c::AbstractMatrix{Bool}, β::Float64, Jx::Float64, Jy::Float64, h::Float64)
    """
  Do one flip (random one single-spin update):
  returns: true if the metropolis move is accepted or false otherwise
    """
    Lx,Ly = size(c)
    nx = rand(1:Lx)
    ny = rand(1:Ly)
    deltaE = - 2*local_energy(c,nx,ny,Jx,Jy,h) #variation
    
    w = exp(-β * deltaE)
    accepted = false
    if ( rand() < w)
        c[nx,ny] = !c[nx,ny] #Flipping spin at nx,ny
        accepted = true
    end
    return accepted
    
end

function cluster_update!(c::AbstractArray{Bool}, beta::Float64, J::Float64)
    """
    Implementing cluster update algorithm
    return: the ratio size of the updated cluster with respect to the system size
    """
    Lx, Ly = size(c)
    N_spins = Lx*Ly
    
    #Choosing a random site
    nx = rand(1:Lx)
    ny = rand(1:Ly)
    
    #Getting the direction of the random spin
    refspin = c[nx,ny] #defining the direction of the potential cluster
    oppspin = !c[nx,ny] #defining the opposite direction
    
    c[nx,ny] = oppspin #flipping the random spin
    
    stack = [(nx,ny)] #List of spins to be visited
    num_tobevisited = 1
    
    size_cluster = 1 #size of the cluster
    
    while num_tobevisited > 0 #as long as the number of spins to be visited is > 0
        current_spin = stack[num_tobevisited] #current spin is the last spin in the stack array
        pop!(stack)
        num_tobevisited -= 1
        
        nns = neighbours(current_spin[1], current_spin[2], Lx, Ly)
        for (a,b) in nns
            if c[a,b] == refspin && rand() < (1-exp(-2*beta*J))
                push!(stack, (a,b))
                num_tobevisited += 1
                c[a,b] = oppspin 
                size_cluster +=1
            end
        end
    end 
    return size_cluster/N_spins
end

function sampledata(Lx::Int64, Ly::Int64,n_sweeps::Int64, β::Float64, Jx::Float64, Jy::Float64, h::Float64; c_init = nothing, meas_func = nothing)
        #sampledata that takes the following as arguments
        # Lx, Ly: dimensions of system
        #n_sweeps: number of sweeps
        #β: 1/Temperature
        #
        # Jx, Jy: couplings
        #h: external field strength
        # c_init: BitArray representing initial configuration
        # meas_func: Function that calculates observable for each config
        #---------------------------------------------------------------

        """
            returns measured data array and acceptance ratio
        """
        if c_init == nothing
            c_init = bitrand(Lx,Ly)
        end
        if (Lx, Ly) != size(c_init)
            @error "Wrong configuration size or choice"
        end
        #work on copy of initial configuration
        config = copy(c_init)
        N = Lx*Ly
        data_list = zeros(n_sweeps)
        accept_ratio = 0.0
        # n_sweeps of N flips (one sweep)
        for i  in 1:n_sweeps
            for j in 1:N
                bool = update!(config, β, Jx, Jy, h)
                if bool
                    accept_ratio += 1.0
                end
            end
            if meas_func == nothing
                observable = NaN
            else
                observable = meas_func(config, Jx, Jy, h)
            end
            data_list[i] = observable
        end
        accept_ratio = accept_ratio/(N*n_sweeps)
        return data_list, accept_ratio
end

function sampledata_clusterupdate(Lx::Int64, Ly::Int64,n_sweeps::Int64, β::Float64, J::Float64; c_init = nothing, meas_func = nothing)
        #sampledata that takes the following as arguments
        # Lx, Ly: dimensions of system
        #n_sweeps: number of sweeps
        #β: 1/Temperature
        # J: Isotropic coupling
        # c_init: BitArray representing initial configuration
        # meas_func: Function that calculates observable for each config
        #---------------------------------------------------------------
        """
            returns measured data array and mean ratio of the clusters with respect to the system size
        """
        if c_init == nothing
            c_init = bitrand(Lx,Ly)
        end
        if (Lx, Ly) != size(c_init)
            @error "Wrong configuration size or choice"
        end
        #work on copy of initial configuration
        config = copy(c_init)
        N = Lx*Ly
        data_list = zeros(n_sweeps)
        size_ratio = 0.0 #ratio size of cluster
        for i  in 1:n_sweeps
            size_ratio += cluster_update!(config, β, J)
            
            if meas_func == nothing
                observable = NaN
            else
                observable = meas_func(config, J, J, 0.0)
            end
            data_list[i] = observable
        end
        size_ratio = size_ratio/n_sweeps
        return data_list, size_ratio
end

function make_bins(v::Array{T,1}, bin_length::Int64) where T<:Number
    
    nbins = length(v)÷bin_length
    output = zeros(nbins)
    
    for i in 1:nbins
        output[i] = mean(v[((i-1)*bin_length +1) : i*bin_length])
    end
        
    return output
end

function meas_stat(v_binned::Array{T,1}) where T <: Number
    
    μ = mean(v_binned)
    v = var(v_binned)/length(v_binned)
    
    return (μ, sqrt(v))
end

function mean_stdev_vs_binlength(data::AbstractVector{Float64}, length_bin_array::AbstractVector{Int})
    """
    this function takes in a raw data array and array for the values of bin lengths.
    For each bin length it measures the statistics and adds to them to arrays which are them plotted vs bin lengths.  
    """
    
    meanarr=[]
    stdevarr=[]
    
    for n in length_bin_array
        tnbin=length(data)÷n
        databinned = make_bins(data,n)
        mean, stdev = meas_stat(databinned)
        push!(meanarr,mean) 
        push!(stdevarr,stdev)
    end
    
    Plots.plot(length_bin_array, meanarr, yerror=stdevarr, legend=false)
    xlabel!("bin length")
    ylabel!("m")
    title!("Mean magnetization vs bin length")
    
end

function discrete_hist(raw_data::Array{T,1}, sorted::Bool = false) where T <: Number
    
    d = proportionmap(raw_data)
    
    x = collect(keys(d))
    y = collect(values(d))
    
    if sorted
        ind_sort = sortperm(x)
        return (x[ind_sort], y[ind_sort])
    else
        return (x, y)
    end

end

function neighbours_replica(nx::Int,ny::Int,Lx::Int,Ly::Int,l::Int)
    #new neighbour function for replica topology
    #Lx and Ly is length of a single system
    #l is the length of subsystem
    if nx<=l
        return ((mod1(nx+1, Lx), ny), (mod1(nx-1, Lx), ny), 
    (nx, mod1(ny+1, 2*Ly)), (nx, mod1(ny-1, 2*Ly)))
    elseif ny<=Ly
        return ((mod1(nx+1, Lx), ny), (mod1(nx-1, Lx), ny), 
    (nx, mod1(ny+1, Ly)), (nx, mod1(ny-1, Ly)))
    else 
        return ((mod1(nx+1, Lx), ny), (mod1(nx-1, Lx), ny), 
    (nx, mod1(ny-Ly+1, Ly)+Ly), (nx, mod1(ny-Ly-1, Ly)+Ly))
    end
end

end
