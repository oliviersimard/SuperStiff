using SuperStiff.PeriodizeSC
using SuperStiff.Stiffness


macro assertion(ex, text)
    :($ex ? nothing : error("Assertion failed: ", $(text)))
end

# Reading parameters from JSON file
paramsfile = "params.json"
params = JSON.parsefile(paramsfile)
println(typeof(params))

# Defining own dictionnary for convenience
block_params = Dict{String,Any}(
	"path_to_files" => params["path_to_files"],
	"data_loop" => params["data_loop"],
	"pattern" => params["pattern"], # Pattern of the files to be loaded
	"beta" => params["beta"],
	"print_mu_dop" => params["Print_mu_dop"], # Parameter used to create stiffness.dat file (file used with SE_G_fig_producer.py)
	"w_discretization" => params["w_discretization"], # Matsubara frequency grid
	"AFM_SC_NOCOEX" => params["AFM_SC_NOCOEX"],
	"cumulant" => params["cumulant"],
	"Periodization" => params["Periodization"],
	"fout_name" => params["fout_name"]
	)

# Check if appropriate input was entered
if block_params["cumulant"] in [0,1]; nothing else; throw(ErrorException("Cumulant takes in 0 or 1 as arguments")) end
if block_params["Periodization"] in [0,1]; nothing else; throw(ErrorException("Periodization takes in 0 or 1 as arguments")) end
if block_params["AFM_SC_NOCOEX"] in [0,1]; nothing else; throw(ErrorException("AFM_SC_NOCOEX takes in 0 or 1 as arguments")) end

# Check out if appriopriate inputs entered (Debugging) 
match_data_loop = match(r"^(.*?)(?=/)",block_params["data_loop"]).match
match_data_loop_arr = Array{String,1}([match_data_loop])
M_tol = -0.01

# Decide the name of the file to be created
fout_name = block_params["fout_name"] 

super_data_M, super_datap, list_of_files = Stiffness.read_data_loop(block_params,block_params["print_mu_dop"],block_params["pattern"])

@assertion(length(list_of_files) != 0, "Length of list_of_files must differ from 0. Check if the path_to_files variable has been given the right path!")
@assertion(length(super_datap) != 0, "Length of super_datap must differ from 0. Check your file containing the chemical potential and the doping and make sure they are respectively marked \'mu\' and \'ave_mu\' in the header")

if Stiffness.verbose > 0
    println("Length list_of_files is : ", length(list_of_files),"\n")
    println("Size super_datap is: ", size(super_datap),"\n")
end

length(list_of_files) != length(super_datap[:,1]) && throw(Stiffness.File_Input_Error(:list_file))
list_of_files_and_datap = hcat(list_of_files,super_datap)

# Building list_modulevec defining the instance of modulevec
super_list_modulevec = Stiffness.gen_modulevec_args(block_params,list_of_files,super_datap[:,1],params)

if Stiffness.verbose > 0
    println("typeof list_modulevec: ",typeof(super_list_modulevec))
end

println(Stiffness.ts(super_list_modulevec[1]))
if Stiffness.verbose > 0
    println("Size of sEvec_c in StiffnessArray: ", size(super_list_modulevec[1].data_), " | typeof StiffnessArray: ", typeof(super_list_modulevec[1]))
end

########################################### Beginning of the superfluid stiffness calculations ###########################################
# Computing the superfluid stiffness
#@time list_stiff, list_kIntegral_DOS, list_kIntegral_DOS_nocoex = Stiffness.calc_stiff_funct(super_data_M, super_list_modulevec, match_data_loop_arr, M_tol)

beta = super_list_modulevec[1].vals_["beta"]
println("beta = $(@sprintf("%.2f", beta))")
w_discretization = super_list_modulevec[1].vals_["w_discretization"]
zvec = 1.0im*[(2*n+1)*pi/beta for n in 1:w_discretization]

list_stiff = Array{Complex{Float64},1}()
@time for (l,modulevec_el) in enumerate(super_list_modulevec)
    arr_w_kIntegrated = Matrix{Complex{Float64}}(0,0)
    println("Iteration number: ",l)
    if match_data_loop_arr[1] == "COEX"
        super_data_M_el = super_data_M[l]
    end
    println("Treating $(match_data_loop_arr[1]) data")
    zvec = 1.0im*[(2*n+1)*pi/beta for n in 1:w_discretization] 
    modelvec = PeriodizeSC.ModelVector(Stiffness.ts(modulevec_el)[1],Stiffness.ts(modulevec_el)[2],Stiffness.ts(modulevec_el)[3],Stiffness.ts(modulevec_el)[4],zvec[1:w_discretization], modulevec_el.data_[1,1:w_discretization,:,:])
    arr_w_kIntegrated = Matrix{Complex{Float64}}(0,0)
    if match_data_loop_arr[1] == "COEX"
        @time arr_w_kIntegrated = Stiffness.calc_stiff_funct_COEX(super_data_M_el, modelvec, modulevec_el, M_tol)  
    elseif match_data_loop_arr[1] == "NOCOEX"
        @time arr_w_kIntegrated = Stiffness.calc_stiff_funct_NOCOEX(modelvec, modulevec_el)
    end
    dop_stiff = 2.0/beta*sum(arr_w_kIntegrated[:,2])
    f = open(block_params["fout_name"], "a")
    if l == 1
        write(f, "#M_tol "*"$(M_tol)"*"\n")
    end
    write(f, "$(super_datap[:,1][l])"*"\t\t"*"$(super_datap[:,2][l])"*"\t\t"*"$(real(dop_stiff))"*"\n")
    close(f)
    push!(list_stiff,dop_stiff)
    println(dop_stiff)
end
println("Superfluid Stiffness: ", list_stiff)

println("Program exited")
