module swaperator

using Random

spin(b::Bool) = Float64(2 * b - 1)

function neighbours_doublecopy(nx::Int64,ny::Int64,Lx::Int64,Ly::Int64, l::Int64)
    #new neighbour function for double copy topology
    #Lx and Ly is length of a single copy
    if ny<=Ly
        return ((mod1(nx+1, Lx), ny), (mod1(nx-1, Lx), ny), (nx, mod1(ny+1, Ly)), (nx, mod1(ny-1, Ly)))
    else
        return ((mod1(nx+1, Lx), ny), (mod1(nx-1, Lx), ny), (nx, mod1(ny+1, Ly)+Ly), (nx, mod1(ny-1, Ly)+Ly))
    end 
end

function neighbours_replica(nx::Int,ny::Int,Lx::Int,Ly::Int,l::Int)
    #new neighbour function for replica topology
    #Lx and Ly is length of a single copy
    #l is the length of subsystem
    if nx<=l
        return ((mod1(nx+1, Lx), ny), (mod1(nx-1, Lx), ny), 
    (nx, mod1(ny+1, 2*Ly)), (nx, mod1(ny-1, 2*Ly)))
    elseif ny<=Ly
        return ((mod1(nx+1, Lx), ny), (mod1(nx-1, Lx), ny), 
    (nx, mod1(ny+1, Ly)), (nx, mod1(ny-1, Ly)))
    else 
        return ((mod1(nx+1, Lx), ny), (mod1(nx-1, Lx), ny), 
    (nx, mod1(ny+1, Ly)+Ly), (nx, mod1(ny-1, Ly)+Ly))
    end
end

function local_ratio(c::AbstractArray{Bool}, J::Float64, beta::Float64, l::Int64)
    # c : a bitarray representing the spin configuration of the double copy
    # J: isotropic spin-spin coupling
    # beta : inverse temperature
    # l : size of the subsystem A
        """
        returns the ratio W(l+1)/W(l)
        """
    Lx,Ly = size(c)
    Ly = floor(Int, Ly/2)
    E_top = -J*(spin(c[l+1, Ly])*spin(c[l+1, Ly+1]) + spin(c[l+1, 1])*spin(c[l+1, 2*Ly]))
    E_bottom =  -J*(spin(c[l+1, 1])*spin(c[l+1, Ly]) + spin(c[l+1, Ly+1])*spin(c[l+1, 2*Ly]))
    
    return exp(-beta*(E_top-E_bottom))
end

function ratio(c::AbstractArray{Bool}, J::Float64, beta::Float64, l::Int64, delta::Int64)
        """
        returns the ratio W(l+delta)/W(l)
        """
        ratio = 1.0
        for i=l:l+delta-1
        ratio *= local_ratio(c, J, beta, i)
        end
        return ratio
end


function swaperator_samples(Lx::Int64, Ly::Int64,n_sweeps::Int64, beta::Float64, J::Float64, l::Int64, delta::Int64; c_init = nothing, meas_func = nothing, neighbours_topology = nothing)
        #sampledata that takes the following as arguments
        # Lx, Ly: dimensions of the double copy (Lx,Ly/2 are then the dimensions of each copy)
        # n_sweeps: number of sweeps
        # β: 1/Temperature
        #
        # J: Isotropic couplings
        # l : size of the subregion A 
        # c_init: BitArray representing initial configuration
        # meas_func: Function that calculates observable for each config
        #---------------------------------------------------------------
        """
            returns measured data array and mean ratio of the clusters with respect to the system size using the swaperator method
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
        stack = Tuple{Int, Int}[]
        for i  in 1:n_sweeps
            size_ratio += cluster_update!(config, stack, beta, J, l) #cluster_update can be found in the MonteCarlo module
            
            if meas_func == nothing
                observable = NaN
            else
                observable = meas_func(config, beta, J, l, delta) #an example of such a meas function can be "ratio"
            end
            data_list[i] = observable
        end
        size_ratio = size_ratio/n_sweeps
        return data_list, size_ratio
end


end #end module

# An example to how to use this module to extract Renyi Entropy---------------------------------------------------------------------------
Lx = 20
Ly = 100
beta = 1/2.269
J = 1.0

l = 0
delta = 2
Renyi2 = []
Renyi2_error = []
S = 0.0
S_error = 0.0

n_sweeps = 500000

#A sanity check: the 2-Renyi entropy should be zero for l = 0 and delta = 0
ratios, acceptance_rate = swaperator_samples(Lx, 2*Ly, n_sweeps, beta, J, 0, 0; meas_func = ratio)
R_bin = make_bins(ratios, 100)
mean_R, std_R = meas_stat(R_bin)
println(mean_R) #to monitor progress
S -= log(mean_R)
push!(Renyi2, S)
S_error = sqrt(S_error^2 + std_R^2/mean_R^2) #error propagation through the log
push!(Renyi2_error, S_error)
println(S_error) #to monitor progress

#looping over the subsystem size (incrementing by delta each time)
for l=0:delta:Lx-2
    ratios, acceptance_rate = swaperator_samples(Lx, 2*Ly, n_sweeps, beta, J, l, delta; meas_func = ratio)
    R_bin = make_bins(ratios, 100) # this function can be found in the MonteCarlo module

    mean_R, std_R = meas_stat(R_bin) # this function can be found in the MonteCarlo module
    println(mean_R) #to monitor progress
    S -= log(mean_R)
    push!(Renyi2, S)
    
    S_error = sqrt(S_error^2 + std_R^2/mean_R^2) #error propagation through the log
    println(S_error) #to monitor
    push!(Renyi2_error, S_error)
end

plot(0:2:Lx,Renyi2, yerror=Renyi2_error, ylims=(0,0.65), legend = false, xaxis =  (font(15, "Calibri")), yaxis =  (font(15, "Calibri")))
ylabel!("Renyi Entropy")
xlabel!("Subsystem size")
#End example ------------------------------------------------------------------------------------------------------------------
