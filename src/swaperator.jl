module swaperator

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
            size_ratio += cluster_update!(config, stack, beta, J, l)
            
            if meas_func == nothing
                observable = NaN
            else
                observable = meas_func(config, beta, J, l, delta)
            end
            data_list[i] = observable
        end
        size_ratio = size_ratio/n_sweeps
        return data_list, size_ratio
end


end 
