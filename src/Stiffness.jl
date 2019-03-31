module Stiffness

using NPZ
using Glob
using SuperStiff.PeriodizeSC

###########################################Functions and structures###########################################
global pwd_ = pwd()
global verbose = 0

macro assertion(ex, text)
    :($ex ? nothing : error("Assertion failed: ", $(text)))
end

struct File_Input_Error <: Exception
    var::Symbol
end

# Exception constructor
Base.showerror(io::IO, e::File_Input_Error) = println(io, e.var, " Keys are not balanced (See function read_data_loop)")


# Static type object containing all important parameters
mutable struct StiffnessArray{T, N, S <: AbstractArray, Y <: Associative{String,Any}} <: AbstractArray{T, N}
    # Contains the data array
    data_::S
    # Can have maximum three non-zero hopping parameters
    hopping_val_::Base.RefValue{Tuple{T, T, T, T}}
    # Contains informations about the calculations to do
    vals_::Y
end

# Outer constructor:
StiffnessArray{T,N}(data_::AbstractArray{Complex{T},N}, hopping_val_::Tuple{T,T,T,T}, vals_::Associative{String,Any}) = StiffnessArray{T,N,typeof(data_),typeof(vals_)}(data_, Ref(hopping_val_), vals_)
ts(t::StiffnessArray) = t.hopping_val_[] # Getter
Base.size(A::StiffnessArray) = size(A.data_)
Base.IndexStyle{T<:StiffnessArray}(::Type{T}) = Base.IndexLinear()

function Base.getindex(A::StiffnessArray, I...)
    checkbounds(Bool, A.data_, I...) && return A.data_[I...]
    I... < 1 ? throw(ErrorException("Out of bounds (lower bound)")) : throw(ErrorException("Out of bounds (upper bound)"))
end

"""
Function used to build main dictionnary containing file-related information.

#Argument(s):

- block_params::Dict{String,Any}: Dict-valued argument produced form JSON parsing.
- print_mu_dop::Int64: Binary-valued argument (1 or 0) to decide to print (or not) a file describing chemical potential vs doping
- pattern::String: String-valued argument representing the extension common to the python binary files (*.npy)

#Returns:

- super_datap: 2d array containing chemical potential on first column and doping on second one (Array{Float64,2})
- stiffness.dat: If print_mu_dop = 1, returns a file and exits the program
- list_of_files: list of files from which the self-energies are to be extracted (Array{String,1})
"""
function read_data_loop(block_params::Dict{String,Any}, print_mu_dop::Int64, pattern::String)
    dict_data_file = Dict{String,Any}()
    
    data_file = open(readdlm, block_params["data_loop"])
    data_file_header = data_file[1,:]

    indmu = find(x->x=="mu",data_file_header)
    inddop = find(x->x=="ave_mu",data_file_header)
    indM = find(x->x=="ave_M",data_file_header)

    data_file_mu = data_file[:,indmu]
    data_file_dop = data_file[:,inddop]
    data_file_M = data_file[:,indM]
    length(data_file_mu) != length(data_file_dop) && throw(ErrorException("Length of chemical potential list must be the same length as that of the doping!"))

    data_file_mu = filter(x->x!="mu",data_file_mu)
    data_file_dop = filter(x->x!="ave_mu",data_file_dop)
    data_file_M = filter(x->x!="ave_M",data_file_M)
    datap_h = hcat(data_file_mu,data_file_dop)
    datap_h = convert(Array{Float64,2},datap_h)
    data_file_M = convert(Array{Float64,1},data_file_M)

    list_of_files = glob(string(block_params["path_to_files"],pattern))
    # Exits the program if print_mu_dop = 1
    if print_mu_dop == 1
        writedlm("stiffness.dat", datap_h, "\t\t")
        print_with_color(:red, "Printed stiffness.dat for later use (SE_G_fig_producer.py)\n")
        exit(0)
    else
        nothing
    end
    #println(length(data_file_M), "\n", length(list_datap_h), "\n", length(list_of_files))
    return data_file_M, datap_h, list_of_files
end

"""
Function extracting the last number of the binary filename.

#Argument(s):

- list_of_files::Array{String,1}: 1d array-valued argument containing the filenames (Array{String,1})

#Returns:

- list_num: 1d array-valued output containing the last number of the filenames (Array{Int64,1})
"""
function gen_file_num(list_of_files::Array{String,1})
    list_num = Array{Int64,1}(length(list_of_files))
    for (i,files) in enumerate(list_of_files)
        m = [x for x in eachmatch(r"\d+",files)][end]
        list_num[i] = parse(Int64,m.match)
    end
    if verbose > 0
        println("Length list_num: ", length(list_num))
    end
    return list_num
end

"""
Function producing the arguments defining any instance (to initiate the instance) of type PeriodizeSC.ModelVector.

#Argument(s):

- block_params::Dict{String,Any}: Directory containing a specific set of the input parameters extracted from params.json (Dict{String,Any})
- list_of_files::Array{String,1}: 1d array-valued argument containing the filenames (Array{String,1})
- data_mu::Array{Float64,1}: 1d array containing the chemical potentials (Array{Float64,1})
- params::Dict{String,Any}: Directory containing all of the input parameters extracted from params.json (Dict{String,Any})

#Returns:

- list_modulevec: 1d array containing all the ModelVector objects holding the informations read out of each of the files
"""
function gen_modulevec_args(block_params::Dict{String,Any},list_of_files::Array{String,1},data_mu::Array{Float64,1},params::Dict{String,Any})
    list_modulevec = Array{StiffnessArray{Float64,4,Array{Complex{Float64},4},Dict{String,Any}},1}() #; list_t_mu = Array{Tuple{Float64,Float64,Float64,Float64},1}()
    list_num = gen_file_num(list_of_files)
    zip_num_files = collect(zip(list_num,list_of_files))
    sort!(zip_num_files)
    for (num,files) in zip_num_files
        sEvec_c = npzread(files)
        if verbose > 0
            println(num, " : ", (params["t"],params["tp"],params["tpp"],data_mu[num+1]))
        end
        println("sEvec_c: ", typeof(sEvec_c))
        println("Types of variables: ", typeof(StiffnessArray(sEvec_c, (params["t"],params["tp"],params["tpp"],data_mu[num+1]), block_params)))
        push!(list_modulevec, StiffnessArray(sEvec_c, (params["t"],params["tp"],params["tpp"],data_mu[num+1]), block_params)) ## Pushing the data structure here !!!!!
    end
    return list_modulevec
end

"""
Function calculating the superfluid stiffness calling the module SuperStiff.PeriodizeSC in the regime where there is coexistence.
Function to use (most suitable) when computing the c-axis superfluid stiffness. Can also compute the a- and b-axis superfluid stiffness, but with
performance drawbacks.

#Argument(s):

- super_data_M_el::Float64: Value of the AFM order parameter amplitude (Float64)
- modelvec::PeriodizeSC.ModelVector: ModelVector instance of a particular set of informations read out from a file (PeriodizeSC.ModelVector)
- modulevec_el::Stiffness.StiffnessArray{Float64,4,Array{Complex{Float64},4},Dict{String,Any}}: Data structure to handle the different input parameters extracted from params.json (Stiffness.StiffnessArray{Float64,4,Array{Complex{Float64},4},Dict{String,Any}})
- M_tol: Tolerance of the AFM order parameter amplitude above which AFM is considered to have set in (For debugging purposes only)

#Returns:

- list_stiff: 2d array containing the values of the superfluid stiffness before summing over Matsubara frequencies (Matrix{Complex{Float64}})
"""
function calc_stiff_funct_COEX(super_data_M_el::Float64, modelvec::PeriodizeSC.ModelVector, modulevec_el::Stiffness.StiffnessArray{Float64,4,Array{Complex{Float64},4},Dict{String,Any}}, M_tol::Float64)
    list_stiff = Matrix{Complex{Float64}}(0,0)
    cumulant = modulevec_el.vals_["cumulant"]
    println("c-axis computations")
    if modulevec_el.vals_["Periodization"] == 1
        println("Periodization option set to 1")    
        if cumulant == 0
            cond = abs(super_data_M_el) > M_tol
            if cond
<<<<<<< HEAD
                list_stiff = PeriodizeSC.calcintegral_RBZ(modelvec,PeriodizeSC.make_stiffness_kintegrand_test)
=======
                list_stiff = PeriodizeSC.calcintegral_RBZ(modelvec,PeriodizeSC.make_stiffness_kintegrand_test)  
                #list_stiff = PeriodizeSC.calcintegral_RBZ(modelvec,PeriodizeSC.make_stiffness_kintegrand_all_terms)
>>>>>>> vlighter
            else
                list_stiff = PeriodizeSC.calcintegral_BZ(modelvec,PeriodizeSC.make_stiffness_kintegrand_SC)
            end
            #list_stiff = PeriodizeSC.calcintegral_BZ(modelvec,PeriodizeSC.make_stiffness_kintegrand_test) ## Temporary test <---------------------------------########
        elseif cumulant == 1
            cond = abs(super_data_M_el) > M_tol
            if cond
                list_stiff = PeriodizeSC.calcintegral_RBZ(modelvec,PeriodizeSC.make_stiffness_kintegrand_cum_AFM_SC)
            else
                list_stiff = PeriodizeSC.calcintegral_BZ(modelvec,PeriodizeSC.make_stiffness_kintegrand_cum_SC)
            end
            #list_stiff = PeriodizeSC.calcintegral_BZ(modelvec,PeriodizeSC.make_stiffness_kintegrand_cum_AFM_SC)
        end
    elseif modulevec_el.vals_["Periodization"] == 0
        println("Periodization option set to 0")
        #list_stiff = PeriodizeSC.calcintegral_BZ(modelvec, PeriodizeSC.make_stiffness_cluster_G_kintegrand)
        list_stiff = PeriodizeSC.calcintegral_RBZ(modelvec, PeriodizeSC.make_stiffness_cluster_G_kintegrand_fourth)
    end
    return list_stiff
end

"""
Function calculating the superfluid stiffness calling the module SuperStiff.PeriodizeSC in the regime where there is NO coexistence.
Function to use (most suitable) when computing the c-axis superfluid stiffness. Can also compute the a- and b-axis superfluid stiffness, but with
performance drawbacks.

#Argument(s):

- modelvec::PeriodizeSC.ModelVector: ModelVector instance of a particular set of informations read out from a file (PeriodizeSC.ModelVector)
- modulevec_el::Stiffness.StiffnessArray{Float64,4,Array{Complex{Float64},4},Dict{String,Any}}: Data structure to handle the different input parameters extracted from params.json (Stiffness.StiffnessArray{Float64,4,Array{Complex{Float64},4},Dict{String,Any}})

#Returns:

- list_stiff: 2d array containing the values of the superfluid stiffness before summing over Matsubara frequencies (Matrix{Complex{Float64}})
"""
function calc_stiff_funct_NOCOEX(modelvec::PeriodizeSC.ModelVector, modulevec_el::Stiffness.StiffnessArray{Float64,4,Array{Complex{Float64},4},Dict{String,Any}})
    list_stiff = Matrix{Complex{Float64}}(0,0)
    cumulant = modulevec_el.vals_["cumulant"]
    AFM_SC = modulevec_el.vals_["AFM_SC_NOCOEX"]
    println("abc option set to \"c\"")
    if modulevec_el.vals_["Periodization"] == 1
        println("Periodization option set to 1") 
        if AFM_SC == 0 && cumulant == 0
            list_stiff = PeriodizeSC.calcintegral_BZ(modelvec,PeriodizeSC.make_stiffness_kintegrand_SC)
        elseif AFM_SC == 0 && cumulant == 1
            println("CUM")
            list_stiff = PeriodizeSC.calcintegral_BZ(modelvec,PeriodizeSC.make_stiffness_kintegrand_cum_SC)
        elseif AFM_SC == 1 && cumulant == 0
            println("Per AFM_SC_1")
            list_stiff = PeriodizeSC.calcintegral_BZ(modelvec,PeriodizeSC.make_stiffness_kintegrand_test) 
        elseif AFM_SC == 1 && cumulant == 1
            println("CUM AFM_SC_1")
            list_stiff = PeriodizeSC.calcintegral_BZ(modelvec,PeriodizeSC.make_stiffness_kintegrand_cum_AFM_SC) 
        end
    elseif modulevec_el.vals_["Periodization"] == 0
        println("Tr.")
        println("Periodization option set to 0")
        if (AFM_SC == 0 || AFM_SC == 1)
            list_stiff = PeriodizeSC.calcintegral_BZ(modelvec, PeriodizeSC.make_stiffness_trace_G_kintegrand)
        end
    end
    return list_stiff
end

end ## end of module Stiffness scope
