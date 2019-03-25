using SuperStiff.PeriodizeSC
using NPZ
using Glob

# You have to run convert_converged_green() before running this program properly (produce both selfR and greenR)

filename_to_write = "PER_periodized_zz.dat"

t = 1.0; tpp = 0.0
pwd_ = pwd()
path_ = pwd_*"/"*filename_to_write
Grid_ = 100 # Resolution of k-space that is relevant ONLY for inplane calculations
OPT_ = "PER"  # Options are CUM, PER and TR
AXIS_ = "zz" # Options are xx (a axis), yy (b axis) and zz (c axis)

macro assertion(ex, text)
    :($ex ? nothing : error("Assertion failed: ", $(text)))
end

existing_mu = Array{Float64}(); existing_beta = Array{Float64}(); existing_U = Array{Float64}(); existing_tp = Array{Float64}()
if isfile(path_)
    existing_data = readdlm(path_, header=true)
    existing_tp = existing_data[1][:,2]
    existing_U = existing_data[1][:,3]
    existing_mu = existing_data[1][:,4]
    existing_beta = existing_data[1][:,5]
    #println("length(existing_mu): ", existing_mu, "\n", "length(existing_U): ", existing_U, "\n", "length(existing_beta): ", existing_beta)
    @assertion(length(existing_tp)==length(existing_mu)==length(existing_U)==length(existing_beta), "Length of existing_* differs from each other.") 
else
    println("The file containing the values of the superfluid stiffness is not existing. Will be created!")
end
existing_data = collect(zip(existing_tp,existing_U,existing_mu,existing_beta))
println("existing_data array: ", existing_data)

function grasp_mu(mu_file_path::String)
    data_mu = readdlm(mu_file_path)[:,2][end]
    return data_mu
end

function load_iwn_list(w_file_path)
    data_iwn = readdlm(w_file_path)[:,1]
    return convert(Array{Complex{Float64},1},data_iwn)
end

pattern_SE = r"selfR_aver"
pattern_GE = r"greenR_aver"
list_of_npy_SE_files = Array{String,1}()
list_of_npy_GE_files = Array{String,1}()
for (root, dirs, files) in walkdir(".")
    #println("Files in $root")
    for file in files
        if ismatch(pattern_SE,file)
            push!(list_of_npy_SE_files,joinpath(root, file)) # path to files
        elseif ismatch(pattern_GE,file)
            push!(list_of_npy_GE_files,joinpath(root, file))
        end
    end
end
#println(list_of_npy_GE_files,"\n",list_of_npy_SE_files)

@assertion(length(list_of_npy_SE_files)==length(list_of_npy_GE_files), "Length of list_of_npy_SE_files differs from that of list_of_npy_GE_files.")

regex_exp_beta_n = r"(?<=beta)(\d+\.?\d*[n]\Z)"
regex_exp_beta = r"(?<=beta)(\d+\.?\d*[^n]\Z)"
regex_exp_tp = r"(?<=tp)([-+]\d*\.\d+|\d+)"
regex_exp_U = r"(?<=U)([-+]\d*\.\d+|\d+)"
regex_exp_mu = r"(?<=m)(\d*\.\d+|\d+)"

list_of_npy_GE_files_todo  = Array{String,1}()
list_of_npy_SE_files_todo = Array{String,1}()
list_list_iwn = Array{Array{Complex{Float64},1},1}()
list_N = Array{Float64,1}()
list_params = Array{Tuple{Float64,Float64,Float64,Float64},1}()
for l in 1:length(list_of_npy_SE_files)
    beta = 0.0
    dir_name = dirname(list_of_npy_SE_files[l])
    try
        beta = parse(Float64, match(regex_exp_beta,dir_name).match)
    catch e
        println("Caught the following exception: ", e)
    end
    U = parse(Float64, match(regex_exp_U,dir_name).match)
    mu = parse(Float64, match(regex_exp_mu,dir_name).match)
    tp = parse(Float64, match(regex_exp_tp,dir_name).match)
    tuple_data = ((tp,U,mu,beta),)
    if isempty(find(existing_data.==tuple_data)) && beta != 0.0
        println("tuple_data: ",tuple_data)
        N_files = glob(dir_name*"/*N.dat") # Contains the file containing the chemical potentials
        iwn_files = glob(dir_name*"/*w.dat") # Contains the file containing the Matsubara frequencies
        NN = grasp_mu(N_files...)
        iwn_list = load_iwn_list(iwn_files...)
        push!(list_N,NN)
        push!(list_list_iwn,1.0im*iwn_list) ### Added 1.0im
        push!(list_params,(tp,U,mu,beta))
        push!(list_of_npy_GE_files_todo,list_of_npy_GE_files[l])
        push!(list_of_npy_SE_files_todo,list_of_npy_SE_files[l])
    elseif beta == 0.0
        println("beta is set to: ",0.0)
    else
        println("This folder's data has already been treated!!")
        println("Set already done at index: ",find(existing_data.==tuple_data)...); println()
    end
end

println("list_params: ",list_params)

list_Kintegral_stiff = Array{Tuple{Tuple{Float64,Float64,Float64,Float64},Array{Complex{Float64},2}},1}()
list_Kintegral_DOS = Array{Tuple{Tuple{Float64,Float64,Float64,Float64},Array{Complex{Float64},2}},1}()
list_mu = Array{Float64,1}(); list_U = Array{Float64,1}(); list_beta = Array{Float64,1}(); list_gap = Array{Float64,1}()
list_tp = Array{Float64,1}(); list_N = Array{Float64,1}()
@time for l in 1:1:length(list_of_npy_GE_files_todo)
    dir_name_l = dirname(list_of_npy_GE_files_todo[l])
    SEvec = npzread(list_of_npy_SE_files_todo[l])
    println("SEvec: ",size(SEvec),"\n")
    GEvec = npzread(list_of_npy_GE_files_todo[l])
    println("GEvec: ",size(GEvec),"\n")
    len_iwn_G = size(GEvec)[1]; len_iwn_SE = size(SEvec)[1]
    len_iwn_G != len_iwn_SE && throw(ErrorException("Must have same number of Matsubara frequencies for self-energy and Green's function!"))
    N_data = readdlm(dir_name_l*"/"*"N.dat")
    list_green_npy = glob(dir_name_l*"/greenR*.npy")
    list_green_npy = Array{String,1}([dir_name for dir_name in list_green_npy if !contains(dir_name,"greenR_aver")])
    println("list_green_npy: ",list_green_npy)
    len_N_aver = length(list_green_npy) #Important to know on how many N values one averages (comes down to number of Green's function used to average)
    N_aver = 0.0
    try
        N_aver = sum(N_data[:,2][end-len_N_aver:end])/(len_N_aver+1)
        println("N_aver: ",N_aver,"\t data_N: ",N_data[:,2][end-len_N_aver:end],"\t",len_N_aver,"\n")
    catch e
        if isa(e,BoundsError)
            println("Caught BoundsError! Not enough iterations to average over.")
        else
            println("Unidentified error occured at runtime.")
        end
    end
    params_tuple = list_params[l]
    tp = params_tuple[1]; beta = params_tuple[4]; mu = params_tuple[3]
    anomalous_GE = abs((1/beta)*sum(GEvec[:,1,6]))
    println("anomalous GE: ", anomalous_GE)
    push!(list_gap, anomalous_GE)
    k_array_prebuild = Array{Array{Float64,2},1}()
    tktilde_prebuild = Array{Array{Complex{Float64},2},2}(0,0)
    modelvec = PeriodizeSC.ModelVector(t,tp,tpp,mu,list_list_iwn[l],SEvec)
    cal = Matrix{Complex{Float64}}(0,0)
    if AXIS_ == "zz"
        println("zz")
        if OPT_ == "PER"
            println("PER")
            cal = PeriodizeSC.calcintegral_BZ(modelvec, PeriodizeSC.make_stiffness_kintegrand_SC, len_sEvec_c=len_iwn_G)  ## cal variable exits an array whose size is (length(iw_n),2). The second column holds the superfluid stiffness data.
            println("cal: ", cal)
        elseif OPT_ == "CUM"
            println("CUM")
            cal = PeriodizeSC.calcintegral_BZ(modelvec, PeriodizeSC.make_stiffness_kintegrand_cum_SC, len_sEvec_c=len_iwn_G) ## Idem cal
            println("cal: ", cal)
        elseif OPT_ == "TR"
            println("TR")
            cal = PeriodizeSC.calcintegral_BZ(modelvec, PeriodizeSC.make_stiffness_trace_G_kintegrand, len_sEvec_c=len_iwn_G) ## Idem cal
        end
    elseif AXIS_ == "yy"
        println("yy")
        fct_array = [PeriodizeSC.DyDyEpsilonbark]
        for f in fct_array
            push!(k_array_prebuild, PeriodizeSC.k_grid(PeriodizeSC.Model(modelvec,1),Grid_,f))
        end
        tktilde_prebuild = PeriodizeSC.k_grid(PeriodizeSC.Model(modelvec,1),Grid_,PeriodizeSC.tktilde)
        if OPT_ == "PER"
            println("PER")
            cal = PeriodizeSC.stiffness_NOCOEX_Per_Cum_ab_k_grid(modelvec,k_array_prebuild,tktilde_prebuild,Grid_,0,0,len_sEvec_c=len_iwn_G)
        elseif OPT_ == "CUM"
            println("CUM")
            cal = PeriodizeSC.stiffness_NOCOEX_Per_Cum_ab_k_grid(modelvec,k_array_prebuild,tktilde_prebuild,Grid_,1,0,len_sEvec_c=len_iwn_G)
        end
    elseif AXIS_ == "xx"
        println("xx")
        fct_array = [PeriodizeSC.DxDxEpsilonbark]
        for f in fct_array
            push!(k_array_prebuild, PeriodizeSC.k_grid(PeriodizeSC.Model(modelvec,1),Grid_,f))
        end
        tktilde_prebuild = PeriodizeSC.k_grid(PeriodizeSC.Model(modelvec,1),Grid_,PeriodizeSC.tktilde)
        if OPT_ == "PER"
            println("PER")
            cal = PeriodizeSC.stiffness_NOCOEX_Per_Cum_ab_k_grid(modelvec,k_array_prebuild,tktilde_prebuild,Grid_,0,0,len_sEvec_c=len_iwn_G)
        elseif OPT_ == "CUM"
            println("CUM")
            cal = PeriodizeSC.stiffness_NOCOEX_Per_Cum_ab_k_grid(modelvec,k_array_prebuild,tktilde_prebuild,Grid_,1,0,len_sEvec_c=len_iwn_G)
        end
    end
    push!(list_Kintegral_stiff,(params_tuple,cal))
    push!(list_mu,mu)
    push!(list_U,params_tuple[2])
    push!(list_beta,beta)
    push!(list_tp,tp)
    push!(list_N,N_aver)
end 
println("list_mu: ", list_mu)
stiffness = Array{Float64,1}()
for j in 1:size(list_Kintegral_stiff)[1]
    stiffness_tmp = 0.0
    println(list_Kintegral_stiff[j][1])
    for ii in 1:size(list_Kintegral_stiff[j][2])[1]
        stiffness_tmp+=list_Kintegral_stiff[j][2][ii, 2]
    end
    beta = list_Kintegral_stiff[j][1][4]
    println("beta: ", beta)
    push!(stiffness,(2.0/beta)*real(stiffness_tmp)) #Factor two to sum over all the imaginary axis
    println((2.0/beta)*imag(stiffness_tmp))
end

@assertion(length(list_mu)==length(stiffness), "Length of list_mu differs from that of stiffness.")

params_stiff = hcat(list_N,list_tp,list_U,list_mu,list_beta,stiffness,list_gap)
map!(x->round.(x,8),params_stiff,params_stiff)
println("Stiffness: ", stiffness)
if isfile(path_)
    open(path_,"a") do io
        writedlm(io, params_stiff, "\t\t")
    end
else
    f = open(filename_to_write, "w")
    write(f, "n"*"\t\t"*"tp"*"\t\t"*"U"*"\t\t"*"mu"*"\t\t"*"beta"*"\t\t"*"Stiff"*"\t\t"*"SC_Gap"*"\n")
    writedlm(f, params_stiff, "\t\t")
    close(f)
end
