using SuperStiff.PeriodizeSC
using SuperStiff.Stiffness

paramsfile = "params.json"
params = JSON.parsefile(paramsfile)
Grid_ = 100
M_tol = -0.01
path_ = params["data_loop"]
data = readdlm(path_, header=true)
relevant_data = hcat(data[1][:,find(x->x=="mu",data[2])],data[1][:,find(x->x=="ave_mu",data[2])])

block_params = Dict{String,Any}(
    "path_to_files" => params["path_to_files"],
    "data_loop" => params["data_loop"],
    "pattern" => params["pattern"], # Pattern of the files to be loaded
    "beta" => params["beta"],
    "print_mu_dop" => params["Print_mu_dop"], # Parameter used to create stiffness.dat file (file used with SE_G)
    "w_discretization" => params["w_discretization"], # Matsubara frequency grid
    "AFM_SC_NOCOEX" => params["AFM_SC_NOCOEX"],
    "cumulant" => params["cumulant"],
    "Periodization" => params["Periodization"],
    "fout_name" => params["fout_name"],
    "inplane_axis" => params["inplane_axis"]
    )

super_data_M, super_datap, list_of_files = Stiffness.read_data_loop(block_params,block_params["print_mu_dop"],block_params["pattern"])

#println(length(super_data_M),"\n",super_datap[1],"\n",list_of_files[1],"\n",datap)

list_modulevec = Stiffness.gen_modulevec_args(block_params,list_of_files,super_datap[:,1],params)
println("super_data_M: ", typeof(super_data_M))

match_data_loop = match(r"^(.*?)(?=/)",block_params["data_loop"]).match
println("match_data_loop: ", match_data_loop)

## Choosing the appropriate bare current vertices 
if match_data_loop == "COEX" && block_params["Periodization"] == 0        
    println("Tr.")
    if block_params["inplane_axis"] == "xx"
        fct_array = [PeriodizeSC.DxDxomegak, PeriodizeSC.DxDxzetakomegak, PeriodizeSC.DxDxomegakepsilonk, PeriodizeSC.DxDxzetakepsilonk, PeriodizeSC.DxDxzetak, PeriodizeSC.DxDxzetakepsilonk, PeriodizeSC.DxDxomegakepsilonk, PeriodizeSC.DxDxzetakepsilonk, PeriodizeSC.DxDxepsilonk]
    elseif block_params["inplane_axis"] == "yy"
        fct_array = [PeriodizeSC.DyDyomegak, PeriodizeSC.DyDyzetakomegak, PeriodizeSC.DyDyomegakepsilonk, PeriodizeSC.DyDyzetakepsilonk, PeriodizeSC.DyDyzetak, PeriodizeSC.DyDyzetakepsilonk, PeriodizeSC.DyDyomegakepsilonk, PeriodizeSC.DyDyzetakepsilonk, PeriodizeSC.DyDyepsilonk]
    end
elseif match_data_loop == "COEX" && block_params["Periodization"] == 1
    println("Per. or Cum.")
    if block_params["inplane_axis"] == "xx"
        fct_array = [PeriodizeSC.DxDxEpsilonbark, PeriodizeSC.DxDxZetakbar, PeriodizeSC.DxDxZetakbarepsilonk, PeriodizeSC.DxDxepsilonk]
    elseif block_params["inplane_axis"] == "yy"
        fct_array = [PeriodizeSC.DyDyEpsilonbark, PeriodizeSC.DyDyZetakbar, PeriodizeSC.DyDyZetakbarepsilonk, PeriodizeSC.DyDyepsilonk]
    end
elseif match_data_loop == "NOCOEX" && block_params["Periodization"] == 1 && block_params["AFM_SC_NOCOEX"] == 1
    println("AFM_SC_1")
    if block_params["inplane_axis"] == "xx"
        fct_array = [PeriodizeSC.DxDxZetakbar, PeriodizeSC.DxDxZetakbarepsilonk, PeriodizeSC.DxDxepsilonk]
    elseif block_params["inplane_axis"] == "yy"
        fct_array = [PeriodizeSC.DyDyZetakbar, PeriodizeSC.DyDyZetakbarepsilonk, PeriodizeSC.DyDyepsilonk]
    end
elseif match_data_loop == "NOCOEX" && (block_params["Periodization"] == 1 || block_params["Periodization"] == 0)
    if block_params["inplane_axis"] == "xx"
        println("NOCOEX along xx")
        fct_array = [PeriodizeSC.DxDxEpsilonbark]
    elseif block_params["inplane_axis"] == "yy"
        println("NOCOEX along yy")
        fct_array = [PeriodizeSC.DyDyEpsilonbark]
    end
end

println(fct_array)
k_array_prebuild = Array{Array{Float64,2},1}()
stiff_array = Array{Float64,1}()
tktilde_prebuild = Array{Array{Complex{Float64},2},2}(0,0)
@time for (l,modulevec_el) in enumerate(list_modulevec)
    if match_data_loop == "COEX"
        super_data_M_el = super_data_M[l]
    end
    w_discretization = modulevec_el.vals_["w_discretization"]
    beta = modulevec_el.vals_["beta"]
    zvec = 1.0im*[(2*n+1)*pi/beta for n in 1:w_discretization] 
    modelvec = PeriodizeSC.ModelVector(Stiffness.ts(modulevec_el)[1],Stiffness.ts(modulevec_el)[2],Stiffness.ts(modulevec_el)[3],Stiffness.ts(modulevec_el)[4],zvec[1:w_discretization], modulevec_el.data_[1,1:w_discretization,:,:])
    if l <= 1
        for f in fct_array
            push!(k_array_prebuild, PeriodizeSC.k_grid(PeriodizeSC.Model(modelvec,l),Grid_,f))
        end
        tktilde_prebuild = PeriodizeSC.k_grid(PeriodizeSC.Model(modelvec,l),Grid_,PeriodizeSC.tktilde)
    end
    println("k_array_prebuild: ", typeof(k_array_prebuild))
    println("Iteration number: ",l)
    arr_w_kIntegrated = Matrix{Complex{Float64}}(0,0)
    if match_data_loop == "COEX" && block_params["Periodization"] == 0
        @time arr_w_kIntegrated = PeriodizeSC.stiffness_cluster_G_ab_k_grid(modelvec, k_array_prebuild, tktilde_prebuild, Grid_)    
    elseif match_data_loop == "COEX" && block_params["Periodization"] == 1
        @time arr_w_kIntegrated = PeriodizeSC.stiffness_COEX_Per_Cum_ab_k_grid(modelvec, k_array_prebuild, tktilde_prebuild, Grid_, block_params["cumulant"], super_data_M_el, M_tol)
    elseif match_data_loop == "NOCOEX" && block_params["Periodization"] == 1
        println("NOCOEX Per. or Cum.")
        @time arr_w_kIntegrated = PeriodizeSC.stiffness_NOCOEX_Per_Cum_ab_k_grid(modelvec, k_array_prebuild, tktilde_prebuild, Grid_, block_params["cumulant"], block_params["AFM_SC_NOCOEX"])
    elseif match_data_loop == "NOCOEX" && block_params["Periodization"] == 0
        println("NOCOEX Tr.")
        @time arr_w_kIntegrated = PeriodizeSC.stiffness_NOCOEX_Tr_ab_k_grid(modelvec, k_array_prebuild, tktilde_prebuild, Grid_)
    end
    dop_stiff = 2.0/beta*sum(arr_w_kIntegrated[:,2])
    f = open(params["fout_name"], "a")
    if l == 1
        write(f, "#Grid "*"$(Grid_)"*"\n")
    end
    write(f, "$(relevant_data[:,1][l])"*"\t\t"*"$(relevant_data[:,2][l])"*"\t\t"*"$(real(dop_stiff))"*"\n")
    close(f)
    push!(stiff_array,dop_stiff)
    println(dop_stiff)
end
println("Superfluid Stiffness: ", stiff_array)

