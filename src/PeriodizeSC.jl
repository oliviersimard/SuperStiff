module PeriodizeSC

using Cubature: hcubature
using Cuba
using JSON
using NPZ

_Nc = 4
_NcN = 8 #Sites for nambu = _Nc*2

# Global Variables
II = eye(Complex{Float64}, _Nc)
ZEROS = zeros(Complex{Float64}, _Nc, _Nc)

"""
Type ModelVector holds all the relevant quantities needed for computations. The built-in member function ModelVector() creates an
instance of the type ModelVector.

#Arguments of ModelVector():

- t_::Float64: Nearest-neighbor hopping amplitude (Float64)
- tp_::Float64: Second nearest-neighbor hopping amplitude (Float64)
- tpp_::Float64: Third nearest-neighbor hopping amplitude (Float64)
- mu_::Float64: Chemical potential (Float64)
- wvec_::Float64: Matsubara frequency grid (Array{Complex{Float64}, 1})
- sEvec_c::Array{Complex{Float64}, 3}: Vector of self-energies (Array{Complex{Float64}, 3})
- cumulants_::Array{Complex{Float64}, 3}: Vector of cumulants of the Green's function (Array{Complex{Float64}, 3})

#Returns:

- ModelVector instance.

"""
type ModelVector
   t_::Float64 ; tp_::Float64 ; tpp_::Float64;  mu_::Float64
   wvec_::Array{Complex{Float64}, 1} ; sEvec_c_::Array{Complex{Float64}, 3}
   cumulants_::Array{Complex{Float64}, 3}

   function ModelVector(t::Float64, tp::Float64, tpp::Float64, mu::Float64,
                        wvec::Array{Complex{Float64}, 1}, sEvec_c::Array{Complex{Float64}, 3})

        cumulants = build_cumulants(wvec, mu, sEvec_c)
        return new(t, tp, tpp, mu, wvec, sEvec_c, cumulants)
    end

end

"""
Function member of the type ModelVector returning the cumulant of the Green's function. 
Useful to compute the superfluid stiffness by periodizing the cumulant.

#Arguments:

- wvec_::Array{Complex{Float64}, 1}: Matsubara frequency grid (Array{Complex{Float64}, 1})
- mu_::Float64: Chemical potential (Float64)
- sEvec_c_::Array{Complex{Float64}, 3}: Vector of self-energies (Array{Complex{Float64}, 3})

#Returns:

- cumulants: Cumulants of the Green's function (Complex{Float64} matrix of size 2*Nc X 2*Nc)

"""
function build_cumulants(wvec::Array{Complex{Float64}, 1}, mu::Float64, sEvec_c::Array{Complex{Float64}, 3})

      cumulants = zeros(Complex{Float64}, size(sEvec_c))

      for (ii, ww) in enumerate(wvec)
          tmp = zeros(Complex{Float64}, (_NcN, _NcN))
          tmp[1, 1] = tmp[2, 2] = tmp[3, 3] = tmp[4, 4] = (ww + mu)
          tmp[5, 5] = tmp[6, 6] = tmp[7, 7] = tmp[8, 8] = -conj((ww + mu))
          tmp -= sEvec_c[ii, :, :]
          cumulants[ii, :, :] = inv(tmp)
      end

      return cumulants
end

"""
Type Model holds all the relevant quantities needed for computations for a given Matsubara frequency. The built-in member function Model() instantiates
an object that a priori has a type ModelVector.

#Arguments of Model():

- modelvec: Object of type ModelVector
- ii::Integer: Matsubara frequency index number (Int64)

#Returns:

- Model instance.

"""
type Model
    t_::Float64  ; tp_::Float64 ; tpp_::Float64 ; mu_::Float64
    w_::Complex{Float64} ; sE_::Array{Complex{Float64}, 2}
    cumulant_::Array{Complex{Float64}, 2}

    function Model(modelvec::ModelVector, ii::Integer)
        (t, tp, tpp, mu, w, sE_c, cumulant) = (modelvec.t_, modelvec.tp_, modelvec.tpp_, modelvec.mu_, modelvec.wvec_[ii], modelvec.sEvec_c_[ii, :, :], modelvec.cumulants_[ii, :, :])
        return new(t, tp, tpp, mu, w, sE_c, cumulant)
    end

end

"""
Function to construct the nearest-neighbor dispersion relation on CuO2 square lattice.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- epsilonk: Value of the dispersion relation on the 2d reciprocal lattice

"""
function epsilonk(model::Model, kx::Float64, ky::Float64)
    epsilonk = -2.0*model.t_*(cos(kx) + cos(ky))
    return epsilonk
end

"""
Function to construct the nearest-neighbor dispersion relation derivative with respect to kx on CuO2 square lattice.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- dxepsilonk: Value of the dispersion relation derivative along kx on the 2d reciprocal lattice

"""
function Dxepsilonk(model::Model, kx::Float64, ky::Float64)
    dxepsilonk = 2.0*model.t_*sin(kx)
    return dxepsilonk
end

"""
Function to construct the nearest-neighbor dispersion relation derivative with respect to ky on CuO2 square lattice.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- dyepsilonk: Value of the dispersion relation derivative along ky on the 2d reciprocal lattice

"""
function Dyepsilonk(model::Model, kx::Float64, ky::Float64)
    dyepsilonk = 2.0*model.t_*sin(ky)
    return dyepsilonk
end

"""
Function to construct the second nearest-neighbor dispersion relation on CuO2 square lattice.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- zetak: Value of the dispersion relation on the 2d reciprocal lattice

"""
function zetak(model::Model, kx::Float64, ky::Float64)
    zetak = -2.0*model.tp_*(cos(kx+ky) + cos(kx-ky))
    return zetak
end

"""
Function to construct the second nearest-neighbor dispersion relation derivative along kx on CuO2 square lattice.
 
#Arguments:
 
- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- dxzetak: Value of the dispersion relation derivative along kx on the 2d reciprocal lattice

"""
function Dxzetak(model::Model, kx::Float64, ky::Float64)
    dxzetak = 2.0*model.tp_*(sin(kx+ky)+sin(kx-ky))
    return dxzetak
end

"""
Function to construct the second nearest-neighbor dispersion relation derivative along ky on CuO2 square lattice.
  
#Arguments:
 
- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- dyzetak: Value of the dispersion relation derivative along ky on the 2d reciprocal lattice
 
"""
function Dyzetak(model::Model, kx::Float64, ky::Float64)
    dyzetak = 2.0*model.tp_*(sin(kx+ky)-sin(kx-ky))
    return dyzetak
end

"""
Function to construct the third nearest-neighbor dispersion relation on CuO2 square lattice.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- omegak: Value of the dispersion relation on the 2d reciprocal lattice

"""
function omegak(model::Model, kx::Float64, ky::Float64)
    omegak = -2.0*model.tpp_*(cos(2.0*kx)+cos(2.0*ky))
    return omegak
end

"""
Function to construct the third nearest-neighbor dispersion relation derivative along kx on CuO2 square lattice
 
#Arguments:
 
- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- dxomegak: Value of the dispersion relation derivative along kx on the 2d reciprocal lattice

"""
function Dxomegak(model::Model, kx::Float64, ky::Float64)
    dxomegak = 4.0*model.tpp_*sin(2.0*kx)
    return dxomegak
end

"""
Function to construct the third-nearest-neighbor dispersion relation derivative along ky on CuO2 square lattice
  
#Arguments:
 
- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- dyomegak: Value of the dispersion relation derivative along ky on the 2d reciprocal lattice
 
"""
function Dyomegak(model::Model, kx::Float64, ky::Float64)
    dyomegak = 4.0*model.tpp_*sin(2.0*ky)
    return dyomegak
end

"""
Function to construct the full dispersion relation on CuO2 square lattice

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- eps_0: Value of the full dispersion relation on the 2d reciprocal lattice

"""
function eps_0(model::Model, kx::Float64, ky::Float64)
    return(epsilonk(model,kx,ky)+zetak(model,kx,ky)+omegak(model,kx,ky))
end

############################################################################## Important Note ###########################################################################
## The following functions construct the bare current vertices used when computing the in-plane superfluid stiffness. The situation in which each and every one of the ##
## functions is used is specified.                                                                                                                                     ##
#########################################################################################################################################################################

######## Cumulant and Green's function periodization schemes SC only
function DxDxEpsilonbark(model::Model, kx::Float64, ky::Float64)
    return (Dxepsilonk(model,kx,ky)+Dxzetak(model,kx,ky)+Dxomegak(model,kx,ky))*(Dxepsilonk(model,kx,ky)+Dxzetak(model,kx,ky)+Dxomegak(model,kx,ky))
end

function DyDyEpsilonbark(model::Model, kx::Float64, ky::Float64)
    return (Dyepsilonk(model,kx,ky)+Dyzetak(model,kx,ky)+Dyomegak(model,kx,ky))*(Dyepsilonk(model,kx,ky)+Dyzetak(model,kx,ky)+Dyomegak(model,kx,ky))
end

####### Cumulant and Green's function periodization schemes SC+AFM mixed state
function DxDxZetakbar(model::Model, kx::Float64, ky::Float64)
    return (Dxzetak(model,kx,ky)+Dxomegak(model,kx,ky))*(Dxzetak(model,kx,ky)+Dxomegak(model,kx,ky))
end

function DyDyZetakbar(model::Model, kx::Float64, ky::Float64)
    return (Dyzetak(model,kx,ky)+Dyomegak(model,kx,ky))*(Dyzetak(model,kx,ky)+Dyomegak(model,kx,ky))
end

function DxDxZetakbarepsilonk(model::Model, kx::Float64, ky::Float64)
    return (Dxzetak(model,kx,ky)+Dxomegak(model,kx,ky))*Dxepsilonk(model,kx,ky)
end

function DyDyZetakbarepsilonk(model::Model, kx::Float64, ky::Float64)
    return (Dyzetak(model,kx,ky)+Dyomegak(model,kx,ky))*Dyepsilonk(model,kx,ky)
end

######## Trace scheme for SC+AFM mixed state
function DxDxzetak(model::Model, kx::Float64, ky::Float64) 
    return Dxzetak(model,kx,ky)*Dxzetak(model,kx,ky)
end

function DyDyzetak(model::Model, kx::Float64, ky::Float64)
    return Dyzetak(model,kx,ky)*Dyzetak(model,kx,ky)
end

function DxDxomegak(model::Model, kx::Float64, ky::Float64) 
    return Dxomegak(model,kx,ky)*Dxomegak(model,kx,ky)
end

function DyDyomegak(model::Model, kx::Float64, ky::Float64)
    return Dyomegak(model,kx,ky)*Dyomegak(model,kx,ky)
end

function DxDxepsilonk(model::Model, kx::Float64, ky::Float64)
    return Dxepsilonk(model,kx,ky)*Dxepsilonk(model,kx,ky)
end

function DyDyepsilonk(model::Model, kx::Float64, ky::Float64)
    return Dyepsilonk(model,kx,ky)*Dyepsilonk(model,kx,ky)
end

function DxDxzetakomegak(model::Model, kx::Float64, ky::Float64) 
    return Dxzetak(model,kx,ky)*Dxomegak(model,kx,ky)
end

function DyDyzetakomegak(model::Model, kx::Float64, ky::Float64) 
    return Dyzetak(model,kx,ky)*Dyomegak(model,kx,ky)
end

function DxDxzetakepsilonk(model::Model, kx::Float64, ky::Float64)
    return Dxzetak(model,kx,ky)*Dxepsilonk(model,kx,ky)
end

function DyDyzetakepsilonk(model::Model, kx::Float64, ky::Float64)
    return Dyzetak(model,kx,ky)*Dyepsilonk(model,kx,ky)
end

function DxDxomegakepsilonk(model::Model, kx::Float64, ky::Float64)
    return Dxomegak(model,kx,ky)*Dxepsilonk(model,kx,ky)
end

function DyDyomegakepsilonk(model::Model, kx::Float64, ky::Float64)
    return Dyomegak(model,kx,ky)*Dyepsilonk(model,kx,ky)
end

###################################################################### End of Important Note ################################################################################
#############################################################################################################################################################################

"""
Function defining the hopping between two layers (CuO2 planes) within a unit cell.

#Arguments:

- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- tperp_squared: Value of the bilayer hopping in terms of the in-plane wavevectors
(Represents the current vertices in the current-current correlation function)

"""
function tperp(kx::Float64,ky::Float64)
    coskx = cos(kx)
    cosky = cos(ky)
    tperp = -(coskx-cosky)*(coskx-cosky)
    tperp_squared = tperp*tperp
    return 1/2*tperp_squared  ### In reality, one must add a factor of 1/2 to account of the integral along z-axis.
end

"""
Function adding the proper phases proportional to the reduced supercluster BZ to the intercluster hoppings.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- 2N_c x 2N_c complex-valued matrix t(k^{tilde})
"""
function tktilde(model::Model, kx::Float64, ky::Float64)
    t = model.t_ ; tp = model.tp_; tpp = model.tpp_
    k = [kx, ky]
    r_sites = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    K_sites = pi*deepcopy(r_sites)
    t_array = zeros(Complex{Float64}, (_Nc, _Nc))
    
    for i in 1:_Nc
        for j in 1:_Nc
            for K in K_sites
                t_array[i, j] += 1.0/_Nc * exp(1.0im*dot(K + k, r_sites[i] - r_sites[j])) * eps_0(model, (K + k)...)
            end
        end
    end

    return (vcat(hcat(t_array, ZEROS), hcat(ZEROS, -t_array)))
end

"""
Function building the full cluster Green's function.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements, self-energy and chemical potential
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- gf_ktilde: 2N_c x 2N_c complex-valued matrix G_c(i omega_n,k^{tilde})
"""
function build_gf_ktilde(model::Model, kx::Float64, ky::Float64)
    zz = (model.w_ + model.mu_)*II
    zz_c = -conj(-conj(model.w_) + model.mu_)*II
    tmp_zz = (vcat(hcat(zz, ZEROS), hcat(ZEROS, -conj(zz)))) #### -conj(zz)
    gf_ktilde = inv(tmp_zz - tktilde(model, kx, ky) - model.sE_)
    return gf_ktilde
end

"""
Function pre-computing the full cluster Green's function on the BZ. Useful to speed up in-plane computations.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements, self-energy and chemical potential
- tk_el::Array{Complex{Float64},2}: pre-computed k-tilde-dependent hopping term (t(k^{tilde}))
#Returns:

- gf_ktilde: 2N_c x 2N_c complex-valued matrix G_c(i omega_n,k^{tilde})
"""
function build_gf_ktilde_prebuild(model::Model, tk_el::Array{Complex{Float64},2})
    zz = (model.w_ + model.mu_)*II
    tmp_zz = (vcat(hcat(zz, ZEROS), hcat(ZEROS, -conj(zz))))  #Same thing as zz_c = -conj(-conj(model.w_) + model.mu_)*II
    gf_ktilde = inv(tmp_zz - tk_el - model.sE_)
    return gf_ktilde
end

"""
Function used to periodize the Green's function with SC only.

#Arguments:

- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
- args::Array{Complex{Float64},2}: Lattice Green's function in terms of k_tilde before being periodized (Array{Complex{Float64}, 2})

#Returns:

- nambu_periodized: Periodized lattice Green's function (Complex{Float64}-valued matrix 0.5*Nc X 0.5*Nc)

"""
function periodize_nocoex(kx::Float64, ky::Float64, arg::Array{Complex{Float64},2})
    ex = exp(1.0im*kx)
    ey = exp(1.0im*ky)
    emx = conj(ex)
    emy = conj(ey)
    vk1 = [1.0,-1.0*ex,-1.0*ey,ex*ey]                      #equivalent transformation matrix for Green per. would be:
    vk2 = [-1.0*emx,1.0,emx*ey,-1.0*ey]                    #[1.0,-1.0*ex,-1.0*ey,ex*ey] 
    vk3 = [-1.0*emy,emy*ex,1.0,-1.0*ex]                    #[-1.0*emx,1.0,emx*ey,-1.0*ey]
    vk4 = [emx*emy,-1.0*emy,-1.0*emx,1.0]                  #[-1.0*emy,emy*ex,1.0,-1.0*ex]
    nambu_periodized = zeros(Complex{Float64},(2,2))       #[emx*emy,-1.0*emy,-1.0*emx,1.0]

    gup = arg[1:4,1:4]
    ff = arg[1:4,5:end]
    ffdag = arg[5:end,1:4]
    gdown = arg[5:end,5:end]

    llperiodized = [gup,ff,ffdag,gdown]
    vk = [vk1,vk2,vk3,vk4]

    list_total = []
    for elem in llperiodized
        list_i = []
        summ=0.0
        for i in 1:size(elem)[1]
            for j in 1:size(elem)[2]
                push!(list_i, elem[i,j]*vk[i][j])
            end
        end
        summ=sum(list_i)
        #println("sum = ", summ)
        push!(list_total,summ)
    end

    nambu_periodized[1,1] = list_total[1]
    nambu_periodized[1,2] = list_total[2]
    nambu_periodized[2,1] = list_total[3]
    nambu_periodized[2,2] = list_total[4]

    return 0.25*nambu_periodized
end


"""
Function used to periodize the cumulant with SC only.

#Arguments:

- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
- K::Array{Float64,1}: Array of superlattice reciprocal wavevectors (noted K)
- args::Array{Complex{Float64},2}: Lattice Green's function in terms of k_tilde before being periodized (Array{Complex{Float64}, 2})

#Returns:

- nambu_cum: Periodized lattice Green's function (Complex{Float64}-valued matrix 0.5*Nc X 0.5*Nc)

"""
function periodize_nocoex_cum(kx::Float64, ky::Float64, K::Array{Float64,1}, arg::Array{Complex{Float64},2})   
    nambu_cum = zeros(Complex{Float64},(2,2))
    R = [[0,0],[1,0],[0,1],[1,1]]
    k = [kx,ky]
    gup = arg[1:4,1:4]
    ff = arg[1:4,5:end]
    ffdag = arg[5:end,1:4]
    gdown = arg[5:end,5:end]

    llperiodized = [gup,ff,ffdag,gdown]
    nambu_sum = Array{Complex{Float64},1}()
    for elem in llperiodized
        tot = 0.0
        for (i,R1) in enumerate(R)
            for (j,R2) in enumerate(R)
                tot += exp(-1.0im*dot(k + K + [pi,pi],R1 - R2))*elem[i,j]
            end
        end
        push!(nambu_sum,tot)
    end
    nambu_cum[1,1] = nambu_sum[1]; nambu_cum[1,2] = nambu_sum[2]
    nambu_cum[2,1] = nambu_sum[3]; nambu_cum[2,2] = nambu_sum[4]
    
    return nambu_cum
end

"""
Function associated to the function periodizing the cumulant with SC only. Adds the dispersion relations after having built the cumulant.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements, self-energy and chemical potential
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
- K::Array{Float64,1}: Array of superlattice reciprocal wavevectors (noted K)

#Returns:

- Cumulant of the Green's function (Complex{Float64}-valued matrix Nc X Nc)

"""
function periodize_nocoex_cum_suite(model::Model, kx::Float64, ky::Float64, K::Array{Float64,1})
    nambu_cum_inv = zeros(Complex{Float64},(2,2))

    nambu_cum_inv = inv(periodize_nocoex_cum(kx,ky,K,model.cumulant_))
    
    k = [kx,ky]
    
    nambu_cum_inv[1,1] -= eps_0(model,(k + K + [pi,pi])...); nambu_cum_inv[2,2] += eps_0(model,(k + K + [pi,pi])...)

    return inv(nambu_cum_inv)
end

"""
Function associated to the function periodizing the cumulant with SC only. Last inversion step corresping to the 
full Green's function.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements, self-energy and chemical potential
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- nambu_periodized: Periodized lattice Green's function (Complex{Float64}-valued matrix Nc X Nc)
"""
function periodize_nocoex_cum_finale(model::Model, kx::Float64, ky::Float64)
    nambu_periodized = zeros(Complex{Float64},(2,2))
    Ks = [[0.,0.],[pi,0.],[0.,pi],[pi,pi]]

    for K in Ks
        #println(K)
        nambu_periodized += periodize_nocoex_cum_suite(model,kx,ky,K) 
    end

    return 0.25*nambu_periodized
end

"""
Function to periodize the Green's function with coexistence between AF and SC.

#Arguments:

- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
- args::Array{Complex{Float64},2}: Lattice Green's function in terms of k_tilde before periodized (Array{Complex{Float64}, 2})

#Returns:

- nambu_periodized: Periodized lattice Green's function (Complex{Float64}-valued matrix Nc X Nc)

"""
function periodize_AFM_orb(arg::Array{Complex{Float64},2}, kx::Float64, ky::Float64) ###Appropriate periodization method for AF-SC systems
    R_A = [[0.0,0.0],[1.0,1.0]]
    R_B = [[1.0,0.0],[0.0,1.0]]
    K_x = [[0.0,0.0],[pi,0.0]]; K_y = [[0.0,0.0],[0.0,pi]]
    k = [kx,ky]
    nambu_periodized = zeros(Complex{Float64}, _Nc, _Nc)

    gup = arg[1:4, 1:4]
    ff = arg[1:4, 5:end]
    ffdag = arg[5:end, 1:4]
    gdown = arg[5:end, 5:end]

    llgreen = [gup, ff, ffdag, gdown]
    blocks = zeros(Complex{Float64},2,2,4)
    for (ii,elem) in enumerate(llgreen)
        gAA = [[elem[1,1],elem[1,4]],[elem[4,1],elem[4,4]]]
        gAB = [[elem[1,2],elem[1,3]],[elem[4,2],elem[4,3]]]
        gBA = [[elem[2,1],elem[2,4]],[elem[3,1],elem[3,4]]]
        gBB = [[elem[2,2],elem[2,3]],[elem[3,2],elem[3,3]]]
        summAA = summAB = summBA = summBB = 0.0
        for K in K_y    ####K_x and K_y should be equivalent
            for RA in 1:size(R_A)[1]
                for RAprime in 1:size(R_A)[1]
                    summAA += exp(1.0im*dot(K + k, R_A[RAprime] - R_A[RA]))*gAA[RA][RAprime]
                end
            end
        end
        blocks[1,1,ii] = summAA
        for K in K_y
            for RA in 1:size(R_A)[1]
                for RBprime in 1:size(R_B)[1]
                    summAB += exp(1.0im*dot(K + k, R_B[RBprime] - R_A[RA]))*gAB[RA][RBprime]
                end
            end
        end
        blocks[1,2,ii] = summAB
        for K in K_y
            for RB in 1:size(R_B)[1]
                for RAprime in 1:size(R_A)[1]
                    summBA += exp(1.0im*dot(K + k, R_A[RAprime] - R_B[RB]))*gBA[RB][RAprime]
                end
            end
        end
        blocks[2,1,ii] = summBA
        for K in K_y
            for RB in 1:size(R_B)[1]
                for RBprime in 1:size(R_B)[1]
                    summBB += exp(1.0im*dot(K + k, R_B[RBprime] - R_B[RB]))*gBB[RB][RBprime]
                end
            end
        end
        blocks[2,2,ii] = summBB
    end
    nambu_periodized[1:2,1:2] = blocks[:,:,1]
    nambu_periodized[1:2,3:end] = blocks[:,:,2]
    nambu_periodized[3:end,1:2] = blocks[:,:,3]
    nambu_periodized[3:end,3:end] = blocks[:,:,4]
    
    return(0.5*nambu_periodized)
end

"""
Function to periodize the cumulant in the case of coexistence between AF and SC.

#Arguments:

- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
- args::Array{Complex{Float64},2}: Lattice Green's function in terms of k_tilde before periodized (Array{Complex{Float64}, 2})
- K::Array{Float64,1}: Array of superlattice reciprocal wavevectors (noted K)

#Returns:

- nambu_cum: Periodized lattice Green's function (Complex{Float64}-valued matrix of size Nc x Nc)

"""
function periodize_AFM_orb_cum(arg::Array{Complex{Float64},2}, kx::Float64, ky::Float64, K::Array{Float64,1}) ###Appropriate periodization method for AFM-SC systems
    nambu_cum = zeros(Complex{Float64},4,4)
    R_A = [[0.0,0.0],[1.0,1.0]]
    R_B = [[1.0,0.0],[0.0,1.0]]
    k = [kx,ky]

    gup = arg[1:4, 1:4]
    ff = arg[1:4, 5:end]
    ffdag = arg[5:end, 1:4]
    gdown = arg[5:end, 5:end]

    llgreen = [gup, ff, ffdag, gdown]
    sum_blocks = zeros(Complex{Float64},2,2,4)
    for (ii,elem) in enumerate(llgreen)
        gAA = [[elem[1,1],elem[1,4]],[elem[4,1],elem[4,4]]]
        gAB = [[elem[1,2],elem[1,3]],[elem[4,2],elem[4,3]]]
        gBA = [[elem[2,1],elem[2,4]],[elem[3,1],elem[3,4]]]
        gBB = [[elem[2,2],elem[2,3]],[elem[3,2],elem[3,3]]]
        summAA = summAB = summBA = summBB = 0.0
        for RA in 1:size(R_A)[1]
            for RAprime in 1:size(R_A)[1]
                summAA += exp(1.0im*dot(K + k, R_A[RAprime] - R_A[RA]))*gAA[RA][RAprime]
            end
        end
        sum_blocks[1,1,ii] = summAA
        for RA in 1:size(R_A)[1]
            for RBprime in 1:size(R_B)[1]
                summAB += exp(1.0im*dot(K + k, R_B[RBprime] - R_A[RA]))*gAB[RA][RBprime]
            end
        end
        sum_blocks[1,2,ii] = summAB
        for RB in 1:size(R_B)[1]
            for RAprime in 1:size(R_A)[1]
                summBA += exp(1.0im*dot(K + k, R_A[RAprime] - R_B[RB]))*gBA[RB][RAprime]
            end
        end
        sum_blocks[2,1,ii] = summBA
        for RB in 1:size(R_B)[1]
            for RBprime in 1:size(R_B)[1]
                summBB += exp(1.0im*dot(K + k, R_B[RBprime] - R_B[RB]))*gBB[RB][RBprime]
            end
        end
        sum_blocks[2,2,ii] = summBB
    end
    nambu_cum[1:2,1:2] = sum_blocks[:,:,1]
    nambu_cum[1:2,3:end] = sum_blocks[:,:,2]
    nambu_cum[3:end,1:2] = sum_blocks[:,:,3]
    nambu_cum[3:end,3:end] = sum_blocks[:,:,4]
    
    return nambu_cum
end

"""
Function associated to the function periodizing the cumulant in the case of coexistence between AFM and SC. Adds the 
dispersion relations after having built the cumulant.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements, self-energy and chemical potential
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
- K::Array{Float64,1}: Array of superlattice reciprocal wavevectors (noted K)

#Returns:

- Cumulant of the lattice Green's function (Complex{Float64}-valued matrix of size Nc x Nc)

"""
function periodize_cum_coex_suite(model::Model, kx::Float64, ky::Float64, K::Array{Float64,1})
    nambu_cum_inv = inv(periodize_AFM_orb_cum(model.cumulant_,kx,ky,K))
    k = [kx,ky]
    nambu_cum_inv[1,1]-=zetak(model,(k + K)...)+omegak(model,(k + K)...); nambu_cum_inv[2,2]-=zetak(model,(k + K)...)+omegak(model,(k + K)...); nambu_cum_inv[1,2]-=epsilonk(model,(k + K)...); nambu_cum_inv[2,1]-=epsilonk(model,(k + K)...)
    nambu_cum_inv[3,3]+=zetak(model,(k + K)...)+omegak(model,(k + K)...); nambu_cum_inv[4,4]+=zetak(model,(k + K)...)+omegak(model,(k + K)...); nambu_cum_inv[4,3]+=epsilonk(model,(k + K)...); nambu_cum_inv[3,4]+=epsilonk(model,(k + K)...)

    nambu_periodized_final = inv(nambu_cum_inv)
    return nambu_periodized_final
end

"""
Function associated to the function periodizing the cumulant in the case of coexistence between AF and SC. Last inversion step corresping to the 
full Green's function.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements, self-energy and chemical potential
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- nambu_periodized: Periodized lattice Green's function (Complex{Float64}-valued matrix Nc X Nc)

"""
function periodize_coex_cum_finale(model::Model, kx::Float64, ky::Float64)
    nambu_periodized = zeros(Complex{Float64},(4,4))
    Kx = [[0,0],[pi,0]]; Ky = [[0,0],[0,pi]]

    for K in Kx ## or Ky
        nambu_periodized += periodize_cum_coex_suite(model,kx,ky,K) 
    end

    return 0.5*nambu_periodized
end

"""
Function to compute the superfluid stiffness in the case dSC and AF coexist (with cumulant periodization) perpendicular to the CuO2 cuprates plane 
neglecting the current vertex corrections.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
  
#Returns:

- Superfluid stiffness: Value of the superfluid stiffness perpendicular to the plane for dSC+AF data with cumulant periodization
    
"""
function stiffness_cumulant_AFM_SC(model::Model, kx::Float64, ky::Float64)
    nambu_periodized_final = periodize_coex_cum_finale(model,kx,ky)
    return 1.0*real(tperp(kx,ky)*4.0*(nambu_periodized_final[2,4]*nambu_periodized_final[3,1]+nambu_periodized_final[1,4]*nambu_periodized_final[3,2]+
		         nambu_periodized_final[2,3]*nambu_periodized_final[4,1]+nambu_periodized_final[1,3]*nambu_periodized_final[4,2]))
end

"""
Function to compute the superfluid stiffness in the case there is only dSC (with cumulant periodization) perpendicular to the CuO2 cuprates plane 
neglecting the current vertex corrections.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
  
#Returns:

- Superfluid stiffness: Value of the superfluid stiffness perpendicular to the plane for dSC-only data with cumulant periodization
    
"""
function stiffness_cum_SC(model::Model, kx::Float64, ky::Float64)
    nambu_periodized_final = periodize_nocoex_cum_finale(model,kx,ky)
    return 1.0*real(tperp(kx,ky)*4.0*(nambu_periodized_final[1,2]*nambu_periodized_final[2,1]))
end


"""
Function to compute the superfluid stiffness in the case there is only dSC (with Green periodization) perpendicular to the CuO2 cuprates plane and 
neglecting the current vertex corrections

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
  
#Returns:

- Superfluid stiffness: Value of the superfluid stiffness perpendicular to the ab plane for dSC-only data with Green periodization
   
"""
function stiffness_SC(model::Model, kx::Float64, ky::Float64)
    nambu_periodized = periodize_nocoex(kx, ky, build_gf_ktilde(model, kx, ky)) ###### NEW or not NEW <-------------
    return 1.0*real(4.0*tperp(kx,ky)*(nambu_periodized[1,2]*nambu_periodized[2,1])) ##Removed minus sign
end

"""
Function to compute the superfluid stiffness in the case there is only dSC (with Green periodization) in the CuO2 cuprates plane 
neglecting the current vertex corrections

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
- inplane_axis::String: String-valued argument to specify in-plane axis along which computing the superfluid stiffness (\'xx\' or \'yy\') 
- AFM_SC_1::Int64: Integer value set to 1 to compute pure SC superfluid stiffness using the formula in coexisting AF+dSC regime. Otherwize, let it be 0
  
#Returns:

- Superfluid stiffness: Value of the superfluid stiffness in the ab plane for dSC-only data with Green periodization
   
"""
function stiffness_nocoex_per_ab(model::Model, kx::Float64, ky::Float64, inplane_axis::String, AFM_SC_1::Int64)
    if AFM_SC_1 == 1
        G = periodize_AFM_orb(build_gf_ktilde(model, kx, ky), kx, ky)    
        if inplane_axis == "xx" 
            kxky = [DxDxZetakbar,DxDxZetakbarepsilonk,DxDxepsilonk]
        elseif inplane_axis == "yy"
            kxky = [DyDyZetakbar,DyDyZetakbarepsilonk,DyDyepsilonk]
        else
            println("OUPS!") && throw("inplane_axis parameter takes only two possible values: \"xx\" or \"yy\".")
        end
        return 1.0*real(kxky[1](model,kx,ky)*4.0*(G[1,3]*G[3,1]+G[2,3]*G[3,2]+G[1,4]*G[4,1]+G[2,4]*G[4,2])+
                        kxky[2](model,kx,ky)*2.0*((G[1,3]+G[2,4])*(G[3,2]+G[4,1])+(G[1,4]+G[2,3])*(G[3,1]+G[4,2]))+
                        kxky[2](model,kx,ky)*2.0*((G[1,3]+G[2,4])*(G[3,2]+G[4,1])+(G[1,4]+G[2,3])*(G[3,1]+G[4,2]))+
                        kxky[3](model,kx,ky)*4.0*(G[2,4]*G[3,1]+G[1,4]*G[3,2]+G[2,3]*G[4,1]+G[1,3]*G[4,2]))
    elseif AFM_SC_1 == 0
        G = periodize_nocoex(kx, ky, build_gf_ktilde(model, kx, ky))    
        if inplane_axis == "xx" 
            kxky = DxDxEpsilonbark
        elseif inplane_axis == "yy"
            kxky = DyDyEpsilonbark
        else
            println("OUPS!") && throw("inplane_axis parameter takes only two possible values: \"xx\" or \"yy\".")
        end
        return 1.0*real(kxky(model,kx,ky)*4.0*(G[1,2]*G[2,1]))
    end
end

"""
Function to compute the superfluid stiffness in the case dSC+AF coexist (with Green periodization) perpendicular to the CuO2 cuprates plane and 
neglecting the current vertex corrections

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
  
#Returns:

- Superfluid stiffness: Value of the superfluid stiffness perpendicular to the ab plane for dSC+AF data with Green periodization
   
"""
function stiffness_test(model::Model, kx::Float64, ky::Float64)
    nambu_periodized = periodize_AFM_orb(build_gf_ktilde(model, kx, ky), kx, ky)
    return 1.0*real(4.0*tperp(kx,ky)*(nambu_periodized[2,4]*nambu_periodized[3,1]+nambu_periodized[1,4]*nambu_periodized[3,2]+
		           nambu_periodized[2,3]*nambu_periodized[4,1]+nambu_periodized[1,3]*nambu_periodized[4,2]))
end

"""
Function to compute the superfluid stiffness in the case dSC+AF coexist (with Green periodization) in the CuO2 cuprates plane 
neglecting the current vertex corrections

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
- inplane_axis::String: String-valued argument to specify in-plane axis along which computing the superfluid stiffness (\'xx\' or \'yy\') 
- AFM_SC_1::Int64: Integer value set to 1 to compute pure SC superfluid stiffness using the formula in coexisting AF+dSC regime. Otherwise let it be 0
  
#Returns:

- Superfluid stiffness: Value of the superfluid stiffness in the ab plane for dSC+AF data with Green periodization
   
"""
function stiffness_coex_per_ab(model::Model, kx::Float64, ky::Float64, inplane_axis::String)
    G = periodize_AFM_orb(build_gf_ktilde(model, kx, ky), kx, ky)
    if inplane_axis == "xx"
        kxky = [DxDxEpsilonbark,DxDxZetakbar,DxDxZetakbarepsilonk,DxDxepsilonk]
    elseif inplane_axis == "yy"
        kxky = [DyDyEpsilonbark,DyDyZetakbar,DyDyZetakbarepsilonk,DyDyepsilonk]
    else
        println("OUPS!") && throw("inplane_axis parameter takes only two possible values: \"xx\" or \"yy\".")
    end
    return 1.0*real(kxky[1](model,kx,ky)*4.0*(G[1,3]*G[3,1]+G[2,3]*G[3,2]+G[1,4]*G[4,1]+G[2,4]*G[4,2])+
                    kxky[2](model,kx,ky)*2.0*((G[1,3]+G[2,4])*(G[3,2]+G[4,1])+(G[1,4]+G[2,3])*(G[3,1]+G[4,2]))+
                    kxky[2](model,kx,ky)*2.0*((G[1,3]+G[2,4])*(G[3,2]+G[4,1])+(G[1,4]+G[2,3])*(G[3,1]+G[4,2]))+
                    kxky[3](model,kx,ky)*4.0*(G[2,4]*G[3,1]+G[1,4]*G[3,2]+G[2,3]*G[4,1]+G[1,3]*G[4,2]))
end

"""
Function to compute the superfluid stiffness in the case there is dSC and AF (without periodization) perpendicular to the CuO2 cuprates plane 
neglecting the current vertex corrections

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
  
#Returns:

- Superfluid stiffness: Value of the superfluid stiffness perpendicular to the ab plane for dSC+AF data without periodization (tracing)
   
"""
function stiffness_cluster_G(model::Model, kx::Float64, ky::Float64)
    G = build_gf_ktilde(model,kx,ky)
    return 1.0*real(tperp(kx,ky)*4.0*(sum(G[1:4,5:end]))*(sum(G[5:end,1:4])))
end

"""
Function used to compute the superfluid stiffness in the case there is dSC and AF (without periodization) perpendicular to the CuO2 cuprates plane 
neglecting the current vertex corrections

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
   
#Returns:

- Superfluid stiffness: Value of the superfluid stiffness perpendicular to the ab plane for dSC+AF data without periodization
   
"""
function stiffness_cluster_G_first_neighbor(model::Model, kx::Float64, ky::Float64)
    G = build_gf_ktilde(model,kx,ky)
    return 1.0*real(tperp(kx,ky)*4.0*((G[3,7]+G[3,8]+G[4,7]+G[4,8])*(G[5,1]+G[5,2]+G[6,1]+G[6,2])+(G[1,7]+G[1,8]+
                    G[2,7]+G[2,8])*(G[5,3]+G[5,4]+G[6,3]+G[6,4])+(G[3,5]+G[3,6]+G[4,5]+G[4,6])*(G[7,1]+G[7,2]+G[8,1]+G[8,2])+(G[1,5]+G[1,6]+
                    G[2,5]+G[2,6])*(G[7,3]+G[7,4]+G[8,3]+G[8,4])))
end

"""
Function used to compute the superfluid stiffness in the case there is dSC and AF (without periodization) in the CuO2 cuprates plane 
neglecting the current vertex corrections

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
- inplane_axis::String: In-plane axis along which the superfluid stiffness is to be computed.
   
#Returns:

- Superfluid stiffness: Value of the superfluid stiffness in the ab plane for dSC+AF data without periodization
   
"""
function stiffness_cluster_G_ab(model::Model, kx::Float64, ky::Float64, inplane_axis::String)
    G = build_gf_ktilde(model,kx,ky)
    if inplane_axis == "xx"
        kxky = [DxDxomegak, DxDxzetakomegak, DxDxomegakepsilonk, DxDxzetakepsilonk, DxDxzetak, DxDxzetakepsilonk, DxDxomegakepsilonk, DxDxzetakepsilonk, DxDxepsilonk]
    elseif inplane_axis == "yy"
        kxky = [DyDyomegak, DyDyzetakomegak, DyDyomegakepsilonk, DyDyzetakepsilonk, DyDyzetak, DyDyzetakepsilonk, DyDyomegakepsilonk, DyDyzetakepsilonk, DyDyepsilonk]
    else
        println("OUPS!") && throw("inplane_axis parameter takes only two possible values: \"xx\" or \"yy\".")
    end
    return 1.0*real((kxky[1](model,kx,ky)*4.0*(G[1,5]*G[5,1]+G[2,5]*G[5,2]+G[3,5]*G[5,3]+G[4,5]*G[5,4]+G[1,6]*G[6,1]+G[2,6]*G[6,2]+G[3,6]*G[6,3]+G[4,6]*G[6,4]+G[1,7]*G[7,1]+G[2,7]*G[7,2]+G[3,7]*G[7,3]+G[4,7]*G[7,4]+G[1,8]*G[8,1]+G[2,8]*G[8,2]+G[3,8]*G[8,3]+G[4,8]*G[8,4])+
    kxky[2](model,kx,ky)*2.0*((G[1,5]+G[2,6])*(G[5,2]+G[6,1])+(G[1,6]+G[2,5])*(G[5,1]+G[6,2])+(G[3,5]+G[4,6])*(G[5,4]+G[6,3])+(G[3,6]+G[4,5])*(G[5,3]+G[6,4])+(G[1,7]+G[2,8])*(G[7,2]+G[8,1])+(G[1,8]+G[2,7])*(G[7,1]+G[8,2])+(G[3,7]+G[4,8])*(G[7,4]+G[8,3])+(G[3,8]+G[4,7])*(G[7,3]+G[8,4]))+
    kxky[3](model,kx,ky)*2.0*(G[3,5]*G[5,1]+G[4,5]*G[5,1]+G[3,5]*G[5,2]+G[4,5]*G[5,2]+G[3,7]*G[5,3]+G[3,8]*G[5,3]+(G[2,7]+G[2,8])*(G[5,2]+G[6,2])+(G[3,6]+G[4,6])*(G[6,1]+G[6,2])+G[3,7]*G[6,3]+G[3,8]*G[6,3]+(G[4,7]+G[4,8])*(G[5,4]+G[6,4])+G[2,6]*(G[6,3]+G[6,4])+G[3,7]*G[7,1]+G[4,7]*G[7,1]+G[3,7]*G[7,2]+G[4,7]*G[7,2]+G[3,5]*G[7,3]+G[3,6]*G[7,3]+G[4,5]*G[7,4]+G[4,6]*G[7,4]+G[2,7]*(G[7,3]+G[7,4])+G[1,7]*(G[5,1]+G[6,1]+G[7,3]+G[7,4])+G[3,8]*G[8,1]+G[4,8]*G[8,1]+G[1,5]*(G[5,3]+G[5,4]+G[7,1]+G[8,1])+G[1,6]*(G[6,3]+G[6,4]+G[7,1]+G[8,1])+(G[3,8]+G[4,8])*G[8,2]+G[2,6]*(G[7,2]+G[8,2])+G[2,5]*(G[5,3]+G[5,4]+G[7,2]+G[8,2])+G[3,5]*G[8,3]+G[3,6]*G[8,3]+(G[4,5]+G[4,6])*G[8,4]+G[2,8]*(G[8,3]+G[8,4])+G[1,8]*(G[5,1]+G[6,1]+G[8,3]+G[8,4]))+
    kxky[4](model,kx,ky)*2.0*((G[1,5]+G[2,6])*(G[5,2]+G[6,1])+(G[1,6]+G[2,5])*(G[5,1]+G[6,2])+(G[3,5]+G[4,6])*(G[5,4]+G[6,3])+(G[3,6]+G[4,5])*(G[5,3]+G[6,4])+(G[1,7]+G[2,8])*(G[7,2]+G[8,1])+(G[1,8]+G[2,7])*(G[7,1]+G[8,2])+(G[3,7]+G[4,8])*(G[7,4]+G[8,3])+(G[3,8]+G[4,7])*(G[7,3]+G[8,4]))+
    kxky[5](model,kx,ky)*4.0*(G[2,6]*G[5,1]+G[1,6]*G[5,2]+G[4,6]*G[5,3]+G[3,6]*G[5,4]+G[2,5]*G[6,1]+G[1,5]*G[6,2]+G[4,5]*G[6,3]+G[3,5]*G[6,4]+G[2,8]*G[7,1]+G[1,8]*G[7,2]+G[4,8]*G[7,3]+G[3,8]*G[7,4]+G[2,7]*G[8,1]+G[1,7]*G[8,2]+G[4,7]*G[8,3]+G[3,7]*G[8,4])+
    kxky[6](model,kx,ky)*2.0*(G[3,6]*G[5,1]+G[4,6]*G[5,1]+G[3,6]*G[5,2]+G[4,6]*G[5,2]+G[4,7]*G[5,3]+G[4,8]*G[5,3]+G[3,7]*G[5,4]+G[3,8]*G[5,4]+G[3,5]*G[6,1]+G[4,5]*G[6,1]+G[3,5]*G[6,2]+G[4,5]*G[6,2]+G[4,7]*G[6,3]+G[4,8]*G[6,3]+G[3,7]*G[6,4]+G[3,8]*G[6,4]+G[3,8]*G[7,1]+G[4,8]*G[7,1]+G[3,8]*G[7,2]+G[4,8]*G[7,2]+G[4,5]*G[7,3]+G[4,6]*G[7,3]+G[3,5]*G[7,4]+G[3,6]*G[7,4]+G[2,8]*(G[5,1]+G[6,1]+G[7,3]+G[7,4])+G[1,8]*(G[5,2]+G[6,2]+G[7,3]+G[7,4])+G[3,7]*G[8,1]+G[4,7]*G[8,1]+G[2,6]*(G[5,3]+G[5,4]+G[7,1]+G[8,1])+G[2,5]*(G[6,3]+G[6,4]+G[7,1]+G[8,1])+(G[3,7]+G[4,7])*G[8,2]+G[1,6]*(G[5,3]+G[5,4]+G[7,2]+G[8,2])+G[1,5]*(G[6,3]+G[6,4]+G[7,2]+G[8,2])+G[4,5]*G[8,3]+G[4,6]*G[8,3]+(G[3,5]+G[3,6])*G[8,4]+G[2,7]*(G[5,1]+G[6,1]+G[8,3]+G[8,4])+G[1,7]*(G[5,2]+G[6,2]+G[8,3]+G[8,4]))+
    kxky[7](model,kx,ky)*2.0*(G[3,5]*G[5,1]+G[4,5]*G[5,1]+G[3,5]*G[5,2]+G[4,5]*G[5,2]+G[3,7]*G[5,3]+G[3,8]*G[5,3]+(G[2,7]+G[2,8])*(G[5,2]+G[6,2])+(G[3,6]+G[4,6])*(G[6,1]+G[6,2])+G[3,7]*G[6,3]+G[3,8]*G[6,3]+(G[4,7]+G[4,8])*(G[5,4]+G[6,4])+G[2,6]*(G[6,3]+G[6,4])+G[3,7]*G[7,1]+G[4,7]*G[7,1]+G[3,7]*G[7,2]+G[4,7]*G[7,2]+G[3,5]*G[7,3]+G[3,6]*G[7,3]+G[4,5]*G[7,4]+G[4,6]*G[7,4]+G[2,7]*(G[7,3]+G[7,4])+G[1,7]*(G[5,1]+G[6,1]+G[7,3]+G[7,4])+G[3,8]*G[8,1]+G[4,8]*G[8,1]+G[1,5]*(G[5,3]+G[5,4]+G[7,1]+G[8,1])+G[1,6]*(G[6,3]+G[6,4]+G[7,1]+G[8,1])+(G[3,8]+G[4,8])*G[8,2]+G[2,6]*(G[7,2]+G[8,2])+G[2,5]*(G[5,3]+G[5,4]+G[7,2]+G[8,2])+G[3,5]*G[8,3]+G[3,6]*G[8,3]+(G[4,5]+G[4,6])*G[8,4]+G[2,8]*(G[8,3]+G[8,4])+G[1,8]*(G[5,1]+G[6,1]+G[8,3]+G[8,4]))+
    kxky[8](model,kx,ky)*2.0*(G[3,6]*G[5,1]+G[4,6]*G[5,1]+G[3,6]*G[5,2]+G[4,6]*G[5,2]+G[4,7]*G[5,3]+G[4,8]*G[5,3]+G[3,7]*G[5,4]+G[3,8]*G[5,4]+G[3,5]*G[6,1]+G[4,5]*G[6,1]+G[3,5]*G[6,2]+G[4,5]*G[6,2]+G[4,7]*G[6,3]+G[4,8]*G[6,3]+G[3,7]*G[6,4]+G[3,8]*G[6,4]+G[3,8]*G[7,1]+G[4,8]*G[7,1]+G[3,8]*G[7,2]+G[4,8]*G[7,2]+G[4,5]*G[7,3]+G[4,6]*G[7,3]+G[3,5]*G[7,4]+G[3,6]*G[7,4]+G[2,8]*(G[5,1]+G[6,1]+G[7,3]+G[7,4])+G[1,8]*(G[5,2]+G[6,2]+G[7,3]+G[7,4])+G[3,7]*G[8,1]+G[4,7]*G[8,1]+G[2,6]*(G[5,3]+G[5,4]+G[7,1]+G[8,1])+G[2,5]*(G[6,3]+G[6,4]+G[7,1]+G[8,1])+(G[3,7]+G[4,7])*G[8,2]+G[1,6]*(G[5,3]+G[5,4]+G[7,2]+G[8,2])+G[1,5]*(G[6,3]+G[6,4]+G[7,2]+G[8,2])+G[4,5]*G[8,3]+G[4,6]*G[8,3]+(G[3,5]+G[3,6])*G[8,4]+G[2,7]*(G[5,1]+G[6,1]+G[8,3]+G[8,4])+G[1,7]*(G[5,2]+G[6,2]+G[8,3]+G[8,4]))+
    kxky[9](model,kx,ky)*4.0*((G[3,7]+G[3,8]+G[4,7]+G[4,8])*(G[5,1]+G[5,2]+G[6,1]+G[6,2])+(G[1,7]+G[1,8]+G[2,7]+G[2,8])*(G[5,3]+G[5,4]+G[6,3]+G[6,4])+(G[3,5]+G[3,6]+G[4,5]+G[4,6])*(G[7,1]+G[7,2]+G[8,1]+G[8,2])+(G[1,5]+G[1,6]+G[2,5]+G[2,6])*(G[7,3]+G[7,4]+G[8,3]+G[8,4]))))
end

"""
Function to compute the superfluid stiffness in the case there is dSC only (without periodization) perpendicular to the CuO2 cuprates plane 
neglecting the current vertex corrections
 
#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction
   
#Returns:

- Superfluid stiffness: Value of the superfluid stiffness perpendicular to the ab plane for dSC-only data without periodization

"""
function stiffness_trace_G(model::Model, kx::Float64, ky::Float64)
    G = build_gf_ktilde(model,kx,ky)
    return 1.0*real(tperp(kx,ky)*4.0*trace(G[1:4,5:end]*G[5:end,1:4]))
end

"""
Function to compute the superfluid stiffness in the case there is dSC only (without periodization) in the CuO2 cuprates plane neglecting the current vertex corrections

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- kx::Float64: Wavevector in x direction
- ky::Float64: Wavevector in y direction

#Returns:

- Superfluid stiffness: Value of the superfluid stiffness in the ab plane for dSC-only data without periodization

"""
function stiffness_trace_G_ab(model::Model, kx::Float64, ky::Float64, inplane_axis::String)
    G = build_gf_ktilde(model,kx,ky)
    if inplane_axis == "xx"
        kxky = DxDxEpsilonbark
    elseif inplane_axis == "yy"
        kxky = DyDyEpsilonbark
    else
        println("OUPS!") && throw("inplane_axis parameter takes only two possible values: \"xx\" or \"yy\".")
    end
    return 1.0*real(kxky(model,kx,ky)*kxky(model,kx,ky)*4.0*trace(G[1:4,5:end]*G[5:end,1:4]))
end

"""
Template function useful to perform integration over the BZ when computing in-plane superfluid stiffness without periodization. 
Called in module Stiffness.jl.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- param::Tuple{String,String}: Tuple of two strings specifying the direction along which to compute the superfluid stiffness and the state (pure or mixed).

#Returns:

- Superfluid stiffness integrand. It is a function object used to compute integrals.

"""
function make_stiffness_cluster_G_kintegrand_ab(model::PeriodizeSC.Model, param::T) where T <: Tuple{AbstractString,AbstractString}
    function stiffness_kintegrand(kk::Array{Float64,1})
        if isa(param,Tuple{String,String})
            if param[1] == "xx" && param[2] == "COEX"
                return(PeriodizeSC.stiffness_cluster_G_ab(model,kk[1],kk[2],"xx"))
            elseif param[1] == "yy" && param[2] == "COEX"
                return(PeriodizeSC.stiffness_cluster_G_ab(model,kk[1],kk[2],"yy"))
            elseif param[1] == "xx" && param[2] == "NOCOEX"
                return(PeriodizeSC.stiffness_trace_G_ab(model,kk[1],kk[2],"xx"))
            elseif param[1] == "yy" && param[2] == "NOCOEX"
                return(PeriodizeSC.stiffness_trace_G_ab(model,kk[1],kk[2],"yy"))
            end
        else 
            throw(ErrorException("Error occurred in function PeriodizeSC.make_stiffness_cluster_G_kintegrand_ab. Type not permitted in template."))
        end
    end
    return stiffness_kintegrand
end

"""
Template function useful to perform integration over the BZ when computing in-plane superfluid stiffness with periodization. 
Called in module Stiffness.jl.

#Arguments:

- model: model instance to have access to the attributes such as hopping elements
- param::Tuple{String,String,Int64}: Tuple of three elements specifying the direction along which to compute the superfluid stiffness, the state (pure or mixed) and an internal debug parameter.

#Returns:

- Superfluid stiffness integrand. It is a function object used to compute integrals.

"""
function make_stiffness_kintegrand_per_ab(model::PeriodizeSC.Model, param::T) where T <: Tuple{AbstractString,AbstractString,Number}
    function stiffness_kintegrand(kk::Array{Float64,1})
        if isa(param,Tuple{String,String,Int64})
            if param[1] == "xx" && param[2] == "COEX"
                return(PeriodizeSC.stiffness_coex_per_ab(model,kk[1],kk[2],"xx"))
            elseif param[1] == "yy" && param[2] == "COEX"
                return(PeriodizeSC.stiffness_coex_per_ab(model,kk[1],kk[2],"yy"))
            elseif param[1] == "xx" && param[2] == "NOCOEX" && param[3] == 0
                return(PeriodizeSC.stiffness_nocoex_per_ab(model,kk[1],kk[2],"xx",0))
            elseif param[1] == "xx" && param[2] == "NOCOEX" && param[3] == 1
                return(PeriodizeSC.stiffness_nocoex_per_ab(model,kk[1],kk[2],"xx",1))
            elseif param[1] == "yy" && param[2] == "NOCOEX" && param[3] == 0
                return(PeriodizeSC.stiffness_nocoex_per_ab(model,kk[1],kk[2],"yy",0))
            elseif param[1] == "yy" && param[2] == "NOCOEX" && param[3] == 1
                return(PeriodizeSC.stiffness_nocoex_per_ab(model,kk[1],kk[2],"yy",1))
            end
        else 
            throw(ErrorException("Error occurred in function PeriodizeSC.make_stiffness_cluster_G_kintegrand_ab. Type not permitted in template."))
        end
    end
    return stiffness_kintegrand
end

############################################################################## Important Note ###########################################################################
## The following functions act as decorators to the superfluid stiffness functions. These functions are necessary to compute the integrals over k-space. The situation ##
## in which each and every one of the functions is used is specified by its name.                                                                                                  ##
#########################################################################################################################################################################

function make_stiffness_trace_G_kintegrand(model::PeriodizeSC.Model)
    function stiffness_kintegrand(kk::Array{Float64,1})
        return(PeriodizeSC.stiffness_trace_G(model,kk[1],kk[2]))
    end
    return stiffness_kintegrand
end

function make_stiffness_cluster_G_kintegrand(model::PeriodizeSC.Model)
    function stiffness_kintegrand(kk::Array{Float64,1})
        return(PeriodizeSC.stiffness_cluster_G(model,kk[1],kk[2]))
    end
    return stiffness_kintegrand
end

function make_stiffness_cluster_G_kintegrand_fourth(model::PeriodizeSC.Model)
    function stiffness_kintegrand(kk::Array{Float64,1})
        return(PeriodizeSC.stiffness_cluster_G_first_neighbor(model,kk[1],kk[2]))
    end
    return stiffness_kintegrand
end

function make_stiffness_kintegrand_cum_AFM_SC(model::PeriodizeSC.Model)
    function stiffness_kintegrand(kk::Array{Float64, 1})
        return(PeriodizeSC.stiffness_cumulant_AFM_SC(model, kk[1], kk[2]))
    end
    return stiffness_kintegrand
end

function make_stiffness_kintegrand_cum_SC(model::PeriodizeSC.Model)
    function stiffness_kintegrand(kk::Array{Float64, 1})
        return(PeriodizeSC.stiffness_cum_SC(model, kk[1], kk[2]))
    end
    return stiffness_kintegrand
end

function make_stiffness_kintegrand_SC(model::PeriodizeSC.Model)
    function stiffness_kintegrand(kk::Array{Float64,1})
        return(PeriodizeSC.stiffness_SC(model, kk[1], kk[2]))
    end
    return stiffness_kintegrand
end

function make_stiffness_kintegrand_test(model::PeriodizeSC.Model)
    function stiffness_kintegrand(kk::Array{Float64,1})
        return(PeriodizeSC.stiffness_test(model,kk[1],kk[2]))
    end
    return stiffness_kintegrand
end

###################################################################### End of Important Note ################################################################################
#############################################################################################################################################################################

"""
Function used to sum over a grid of the reduced BZ. 

#Arguments:

- modelvector: modelvector instance
- fct: Function object to be used to sum over the reduced BZ.
- gridK::Int64: k-space grid NxN, where N is set to 100 by default. 
- len_sEvec_c::Int64: Length of the Matsubara frequency grid. Set by default to 500.

#Returns:

- Returns an array of two columns contaning the k-integrated Green's function and the associated Matsubara frequency.

"""
function sum_RBZ(modelvector::ModelVector, fct; gridK::Int64 = 100, len_sEvec_c::Int64=500)
    len_sEvec_c > size(modelvector.sEvec_c_)[1] && throw(ErrorException("You have just exceeded the number of Matsubara frequencies available for calculations!"))
    kx = linspace(-pi,pi,gridK)
    ky = linspace(-pi,pi,gridK)
    println("Length of len_sEvec_c and length of modelvector.sEvec_c_, respectively :", len_sEvec_c, size(modelvector.sEvec_c_)[1])
    result_k_ind = Array{Float64}(len_sEvec_c)
    for n in 1:len_sEvec_c
        stiffness = 0.0
        model = Model(modelvector,n)
        for i in 1:size(ky)[1]
            for j in 1:size(kx)[1]
                stiffness += fct(model,kx[j],ky[i])
            end
        end
        stiffness = 1./(2.0*gridK^2)*stiffness
        result_k_ind[n] = stiffness
        println("stiffness = ", stiffness)
    end
    result_out = hcat(modelvector.wvec_[1:len_sEvec_c], result_k_ind)
    return(result_out)
end

function sum_BZ(modelvector::ModelVector, fct; gridK::Int64 = 100, len_sEvec_c::Int64=500)
    println("in sum_BZ: 2.0*sum_RBZ")
    result_out = 2.0*sum_RBZ(modelvector, fct, gridK=gridK, len_sEvec_c=len_sEvec_c)
    return result_out
end

"""
Function used to integrate over the BZ when computing out-of-plane superfluid stiffness. 

#Arguments:Integer value set to 1 to compute pure SC superfluid stiffness using the formula in coexisting AF+dSC regime. Otherwize, let it be 0

- modelvectInteger value set to 1 to compute pure SC superfluid stiffness using the formula in coexisting AF+dSC regime. Otherwize, let it be 0
- fct: FuncInteger value set to 1 to compute pure SC superfluid stiffness using the formula in coexisting AF+dSC regime. Otherwize, let it be 0
- gridK::InInteger value set to 1 to compute pure SC superfluid stiffness using the formula in coexisting AF+dSC regime. Otherwize, let it be 0
- len_sEvecInteger value set to 1 to compute pure SC superfluid stiffness using the formula in coexisting AF+dSC regime. Otherwize, let it be 0

#Returns:

- Returns an array of two columns contaning the k-integrated Green's function and the associated Matsubara frequency.

"""
function calcintegral_RBZ(modelvector::ModelVector, fct; maxevals::Int64=100_000, len_sEvec_c::Int64=500)
    len_sEvec_c > size(modelvector.sEvec_c_)[1] && throw(ErrorException("You have just exceeded the number of Matsubara frequencies available for calculations!"))
    println("Length of len_sEvec_c and length of modelvector.sEvec_c_, respectively : ", len_sEvec_c, "  ", size(modelvector.sEvec_c_)[1])
    result = Array{Float64,1}(len_sEvec_c)
    println("in calcintegral_RBZ, kwargs")
    for n in 1:len_sEvec_c
        model = Model(modelvector, n)
        #println("w_: ", model.w_)
        result[n] = (1.0/2.0)*(2.0*pi)^(-2.0)*hcubature(fct(model), (-pi,-pi), (pi,pi), reltol=1.49e-8, abstol=1.49e-8, maxevals=maxevals)[1]
        #println("stiff: ", result[n])
    end
    result_out = hcat(modelvector.wvec_[1:len_sEvec_c], result)
    return result_out
end

function calcintegral_BZ(modelvector::ModelVector, fct; maxevals::Int64=100_000, len_sEvec_c::Int64=500)
    println("in calcintegral_BZ: 2.0*calcintegral_RBZ")
    result_out = 2.0*calcintegral_RBZ(modelvector,fct,len_sEvec_c=len_sEvec_c)
    return result_out
end

"""
Function used to integrate over the BZ when computing in-plane superfluid stiffness. 

#Arguments:

- modelvector: modelvector instance
- fct: Function object to be used to integrate over the BZ.
- param::Tuple{String,String,Int64}: Tuple of three elements necessary to provide because of the form of the parameter fct used.
- gridK::Int64: k-space grid NxN, where N is set to 100 by default. 
- len_sEvec_c::Int64: Length of the Matsubara frequency grid. Set by default to 500.

#Returns:

- Returns an array of two columns contaning the k-integrated Green's function and the associated Matsubara frequency.

"""
function calcintegral_RBZ_ab(modelvector::ModelVector, fct, param::T; maxevals::Int64=100_000, len_sEvec_c::Int64=500) where T <: Tuple
    len_sEvec_c > size(modelvector.sEvec_c_)[1] && throw(ErrorException("You have just exceeded the number of Matsubara frequencies available for calculations!"))
    println("Length of len_sEvec_c and length of modelvector.sEvec_c_, respectively : ", len_sEvec_c, "  ",size(modelvector.sEvec_c_)[1])
    result = Array{Float64,1}(len_sEvec_c)
    println("in calcintegral_RBZ_ab, kwargs")
    for n in 1:len_sEvec_c
        model = Model(modelvector, n)
        result[n] = (1.0/2.0)*(2.0*pi)^(-2.0)*hcubature(fct(model,param), (-pi,-pi), (pi,pi), reltol=1.49e-8, abstol=1.49e-8, maxevals=maxevals)[1]
        println("stiff: ", result[n])
    end
    result_out = hcat(modelvector.wvec_[1:len_sEvec_c], result)
    return result_out
end

function calcintegral_BZ_ab(modelvector::ModelVector, fct, param::T; maxevals::Int64=100_000, len_sEvec_c::Int64=500) where T <: Tuple
    println("in calcintegral_BZ_ab: 2.0*calcintegral_RBZ_ab")
    result_out = 2.0*calcintegral_RBZ_ab(modelvector,fct,param,maxevals=maxevals,len_sEvec_c=len_sEvec_c)
    return result_out
end

"""
Function used to map any functions on desired k-space grid. Used when pre-computing current vertices. 

#Arguments:

- model: model instance
- grid::Int64: k-space grid dimension. 
- fct: Function object to be mapped onto the BZ grid.

#Returns:

- Returns an array object of type Array{Float64}(Grid,Grid). This object contains the fct values on the BZ grid.

"""
function k_grid(model::Model,Grid::Int64,fct) #This function can take tktilde or dispersion relation
    try
        BZ_grid = Array{Float64}(Grid,Grid)
        for (i,ky) in enumerate(linspace(-pi,pi,Grid))
            for (j,kx) in enumerate(linspace(-pi,pi,Grid))
                BZ_grid[i,j] = fct(model,kx,ky)
            end
        end
        return BZ_grid
    catch ex
        if isa(ex,MethodError)
            BZ_grid = Array{Array{Complex{Float64},2}}(Grid,Grid)
            for (i,ky) in enumerate(linspace(-pi,pi,Grid))
                for (j,kx) in enumerate(linspace(-pi,pi,Grid))
                    BZ_grid[i,j] = fct(model,kx,ky)
                end
            end
            return BZ_grid
        else
            throw(ErrorException("Problem in definition of the type of BZ_grid. See function PeriodizeSC.k_grid."))
        end
    end
end

"""
Function used to in-plane AF+dSC superfluid stiffness without periodization. 

#Arguments:

- modelvector: modelvector instance
- kxky::Array{Array{Float64,2},1}: Pre-computed current vertices. 
- tk::Array{Array{Complex{Float64},2},2}: Pre-computed hopping matrix represented in the reduced supercluster BZ.
- Grid::Int64: Value of the k-space grid dimension.
- len_sEvec_c::Int64: Length of the Matsubara frequency grid. Set by default to 500.

#Returns:

- Returns an array of two columns contaning the k-integrated Green's function and the associated Matsubara frequency.

"""
function stiffness_cluster_G_ab_k_grid(modelvector::ModelVector, kxky::Array{Array{Float64,2},1}, tk::Array{Array{Complex{Float64},2},2}, Grid::Int64; len_sEvec_c::Int64=500)
    len_sEvec_c > size(modelvector.sEvec_c_)[1] && throw(ErrorException("You have exceeded the number of Matsubara frequencies available for calculations."))
    result_n = Array{Float64,1}(len_sEvec_c)
    for n in 1:len_sEvec_c
        model = Model(modelvector,n) ########################### Should devide by two when tracing ???
        Sum = 0.0
        for (i,ky) in enumerate(linspace(-pi,pi,Grid))
            for (j,kx) in enumerate(linspace(-pi,pi,Grid))
                G = build_gf_ktilde_prebuild(model,tk[i,j])
                Sum += 1.0*(kxky[1][i,j]*4.0*(G[1,5]*G[5,1]+G[2,5]*G[5,2]+G[3,5]*G[5,3]+G[4,5]*G[5,4]+G[1,6]*G[6,1]+G[2,6]*G[6,2]+G[3,6]*G[6,3]+G[4,6]*G[6,4]+G[1,7]*G[7,1]+G[2,7]*G[7,2]+G[3,7]*G[7,3]+G[4,7]*G[7,4]+G[1,8]*G[8,1]+G[2,8]*G[8,2]+G[3,8]*G[8,3]+G[4,8]*G[8,4])+
                kxky[2][i,j]*2.0*((G[1,5]+G[2,6])*(G[5,2]+G[6,1])+(G[1,6]+G[2,5])*(G[5,1]+G[6,2])+(G[3,5]+G[4,6])*(G[5,4]+G[6,3])+(G[3,6]+G[4,5])*(G[5,3]+G[6,4])+(G[1,7]+G[2,8])*(G[7,2]+G[8,1])+(G[1,8]+G[2,7])*(G[7,1]+G[8,2])+(G[3,7]+G[4,8])*(G[7,4]+G[8,3])+(G[3,8]+G[4,7])*(G[7,3]+G[8,4]))+
                kxky[3][i,j]*2.0*(G[3,5]*G[5,1]+G[4,5]*G[5,1]+G[3,5]*G[5,2]+G[4,5]*G[5,2]+G[3,7]*G[5,3]+G[3,8]*G[5,3]+(G[2,7]+G[2,8])*(G[5,2]+G[6,2])+(G[3,6]+G[4,6])*(G[6,1]+G[6,2])+G[3,7]*G[6,3]+G[3,8]*G[6,3]+(G[4,7]+G[4,8])*(G[5,4]+G[6,4])+G[2,6]*(G[6,3]+G[6,4])+G[3,7]*G[7,1]+G[4,7]*G[7,1]+G[3,7]*G[7,2]+G[4,7]*G[7,2]+G[3,5]*G[7,3]+G[3,6]*G[7,3]+G[4,5]*G[7,4]+G[4,6]*G[7,4]+G[2,7]*(G[7,3]+G[7,4])+G[1,7]*(G[5,1]+G[6,1]+G[7,3]+G[7,4])+G[3,8]*G[8,1]+G[4,8]*G[8,1]+G[1,5]*(G[5,3]+G[5,4]+G[7,1]+G[8,1])+G[1,6]*(G[6,3]+G[6,4]+G[7,1]+G[8,1])+(G[3,8]+G[4,8])*G[8,2]+G[2,6]*(G[7,2]+G[8,2])+G[2,5]*(G[5,3]+G[5,4]+G[7,2]+G[8,2])+G[3,5]*G[8,3]+G[3,6]*G[8,3]+(G[4,5]+G[4,6])*G[8,4]+G[2,8]*(G[8,3]+G[8,4])+G[1,8]*(G[5,1]+G[6,1]+G[8,3]+G[8,4]))+
                kxky[4][i,j]*2.0*((G[1,5]+G[2,6])*(G[5,2]+G[6,1])+(G[1,6]+G[2,5])*(G[5,1]+G[6,2])+(G[3,5]+G[4,6])*(G[5,4]+G[6,3])+(G[3,6]+G[4,5])*(G[5,3]+G[6,4])+(G[1,7]+G[2,8])*(G[7,2]+G[8,1])+(G[1,8]+G[2,7])*(G[7,1]+G[8,2])+(G[3,7]+G[4,8])*(G[7,4]+G[8,3])+(G[3,8]+G[4,7])*(G[7,3]+G[8,4]))+
                kxky[5][i,j]*4.0*(G[2,6]*G[5,1]+G[1,6]*G[5,2]+G[4,6]*G[5,3]+G[3,6]*G[5,4]+G[2,5]*G[6,1]+G[1,5]*G[6,2]+G[4,5]*G[6,3]+G[3,5]*G[6,4]+G[2,8]*G[7,1]+G[1,8]*G[7,2]+G[4,8]*G[7,3]+G[3,8]*G[7,4]+G[2,7]*G[8,1]+G[1,7]*G[8,2]+G[4,7]*G[8,3]+G[3,7]*G[8,4])+
                kxky[6][i,j]*2.0*(G[3,6]*G[5,1]+G[4,6]*G[5,1]+G[3,6]*G[5,2]+G[4,6]*G[5,2]+G[4,7]*G[5,3]+G[4,8]*G[5,3]+G[3,7]*G[5,4]+G[3,8]*G[5,4]+G[3,5]*G[6,1]+G[4,5]*G[6,1]+G[3,5]*G[6,2]+G[4,5]*G[6,2]+G[4,7]*G[6,3]+G[4,8]*G[6,3]+G[3,7]*G[6,4]+G[3,8]*G[6,4]+G[3,8]*G[7,1]+G[4,8]*G[7,1]+G[3,8]*G[7,2]+G[4,8]*G[7,2]+G[4,5]*G[7,3]+G[4,6]*G[7,3]+G[3,5]*G[7,4]+G[3,6]*G[7,4]+G[2,8]*(G[5,1]+G[6,1]+G[7,3]+G[7,4])+G[1,8]*(G[5,2]+G[6,2]+G[7,3]+G[7,4])+G[3,7]*G[8,1]+G[4,7]*G[8,1]+G[2,6]*(G[5,3]+G[5,4]+G[7,1]+G[8,1])+G[2,5]*(G[6,3]+G[6,4]+G[7,1]+G[8,1])+(G[3,7]+G[4,7])*G[8,2]+G[1,6]*(G[5,3]+G[5,4]+G[7,2]+G[8,2])+G[1,5]*(G[6,3]+G[6,4]+G[7,2]+G[8,2])+G[4,5]*G[8,3]+G[4,6]*G[8,3]+(G[3,5]+G[3,6])*G[8,4]+G[2,7]*(G[5,1]+G[6,1]+G[8,3]+G[8,4])+G[1,7]*(G[5,2]+G[6,2]+G[8,3]+G[8,4]))+
                kxky[7][i,j]*2.0*(G[3,5]*G[5,1]+G[4,5]*G[5,1]+G[3,5]*G[5,2]+G[4,5]*G[5,2]+G[3,7]*G[5,3]+G[3,8]*G[5,3]+(G[2,7]+G[2,8])*(G[5,2]+G[6,2])+(G[3,6]+G[4,6])*(G[6,1]+G[6,2])+G[3,7]*G[6,3]+G[3,8]*G[6,3]+(G[4,7]+G[4,8])*(G[5,4]+G[6,4])+G[2,6]*(G[6,3]+G[6,4])+G[3,7]*G[7,1]+G[4,7]*G[7,1]+G[3,7]*G[7,2]+G[4,7]*G[7,2]+G[3,5]*G[7,3]+G[3,6]*G[7,3]+G[4,5]*G[7,4]+G[4,6]*G[7,4]+G[2,7]*(G[7,3]+G[7,4])+G[1,7]*(G[5,1]+G[6,1]+G[7,3]+G[7,4])+G[3,8]*G[8,1]+G[4,8]*G[8,1]+G[1,5]*(G[5,3]+G[5,4]+G[7,1]+G[8,1])+G[1,6]*(G[6,3]+G[6,4]+G[7,1]+G[8,1])+(G[3,8]+G[4,8])*G[8,2]+G[2,6]*(G[7,2]+G[8,2])+G[2,5]*(G[5,3]+G[5,4]+G[7,2]+G[8,2])+G[3,5]*G[8,3]+G[3,6]*G[8,3]+(G[4,5]+G[4,6])*G[8,4]+G[2,8]*(G[8,3]+G[8,4])+G[1,8]*(G[5,1]+G[6,1]+G[8,3]+G[8,4]))+
                kxky[8][i,j]*2.0*(G[3,6]*G[5,1]+G[4,6]*G[5,1]+G[3,6]*G[5,2]+G[4,6]*G[5,2]+G[4,7]*G[5,3]+G[4,8]*G[5,3]+G[3,7]*G[5,4]+G[3,8]*G[5,4]+G[3,5]*G[6,1]+G[4,5]*G[6,1]+G[3,5]*G[6,2]+G[4,5]*G[6,2]+G[4,7]*G[6,3]+G[4,8]*G[6,3]+G[3,7]*G[6,4]+G[3,8]*G[6,4]+G[3,8]*G[7,1]+G[4,8]*G[7,1]+G[3,8]*G[7,2]+G[4,8]*G[7,2]+G[4,5]*G[7,3]+G[4,6]*G[7,3]+G[3,5]*G[7,4]+G[3,6]*G[7,4]+G[2,8]*(G[5,1]+G[6,1]+G[7,3]+G[7,4])+G[1,8]*(G[5,2]+G[6,2]+G[7,3]+G[7,4])+G[3,7]*G[8,1]+G[4,7]*G[8,1]+G[2,6]*(G[5,3]+G[5,4]+G[7,1]+G[8,1])+G[2,5]*(G[6,3]+G[6,4]+G[7,1]+G[8,1])+(G[3,7]+G[4,7])*G[8,2]+G[1,6]*(G[5,3]+G[5,4]+G[7,2]+G[8,2])+G[1,5]*(G[6,3]+G[6,4]+G[7,2]+G[8,2])+G[4,5]*G[8,3]+G[4,6]*G[8,3]+(G[3,5]+G[3,6])*G[8,4]+G[2,7]*(G[5,1]+G[6,1]+G[8,3]+G[8,4])+G[1,7]*(G[5,2]+G[6,2]+G[8,3]+G[8,4]))+
                kxky[9][i,j]*4.0*((G[3,7]+G[3,8]+G[4,7]+G[4,8])*(G[5,1]+G[5,2]+G[6,1]+G[6,2])+(G[1,7]+G[1,8]+G[2,7]+G[2,8])*(G[5,3]+G[5,4]+G[6,3]+G[6,4])+(G[3,5]+G[3,6]+G[4,5]+G[4,6])*(G[7,1]+G[7,2]+G[8,1]+G[8,2])+(G[1,5]+G[1,6]+G[2,5]+G[2,6])*(G[7,3]+G[7,4]+G[8,3]+G[8,4])))
            end
        end
        result_n[n] = 2./(Grid^2)*real(Sum) # Factor 2 accounts for the imaginary axis
    end
    result_out = hcat(modelvector.wvec_[1:len_sEvec_c],result_n)
    println(result_out)
    return result_out
end

"""
Function used to in-plane AF+dSC superfluid stiffness with periodization. 

#Arguments:

- modelvector: modelvector instance
- kxky::Array{Array{Float64,2},1}: Pre-computed current vertices. 
- tk::Array{Array{Complex{Float64},2},2}: Pre-computed hopping matrix represented in the reduced supercluster BZ.
- Grid::Int64: Value of the k-space grid dimension.
- Cum::Int64: 1 if the periodization is to be done on the cumulant. 0 if the periodization is to be done on Green's function.
- super_data_M_el::Float64: Value of the AF order parameter amplitude.
- M_tol::Float64: Relevant for debugging purposes only. 
- len_sEvec_c::Int64: Length of the Matsubara frequency grid. Set by default to 500.

#Returns:

- Returns an array of two columns contaning the k-integrated Green's function and the associated Matsubara frequency.

"""
function stiffness_COEX_Per_Cum_ab_k_grid(modelvector::ModelVector, kxky::Array{Array{Float64,2},1}, tk::Array{Array{Complex{Float64},2},2}, Grid::Int64, Cum::Int64, super_data_M_el::Float64, M_tol::Float64; len_sEvec_c::Int64=500)
    len_sEvec_c > size(modelvector.sEvec_c_)[1] && throw(ErrorException("You have exceeded the number of Matsubara frequencies available for calculations."))
    result_n = Array{Float64,1}(len_sEvec_c)
    for n in 1:len_sEvec_c
        model = Model(modelvector,n)
        Sum = 0.0
        cond = abs(super_data_M_el) > M_tol
        if cond
            for (i,ky) in enumerate(linspace(-pi,pi,Grid))
                for (j,kx) in enumerate(linspace(-pi,pi,Grid))
                    if Cum == 0
                        gf_ktilde = build_gf_ktilde_prebuild(model,tk[i,j])
                        G = periodize_AFM_orb(gf_ktilde, kx, ky)
                    elseif Cum == 1
                        G = periodize_coex_cum_finale(model, kx, ky)
                    end
                    Sum += 1.0*(kxky[2][i,j]*4.0*(G[1,3]*G[3,1]+G[2,3]*G[3,2]+G[1,4]*G[4,1]+G[2,4]*G[4,2])+
                    kxky[3][i,j]*2.0*((G[1,3]+G[2,4])*(G[3,2]+G[4,1])+(G[1,4]+G[2,3])*(G[3,1]+G[4,2]))+
                    kxky[3][i,j]*2.0*((G[1,3]+G[2,4])*(G[3,2]+G[4,1])+(G[1,4]+G[2,3])*(G[3,1]+G[4,2]))+
                    kxky[4][i,j]*4.0*(G[2,4]*G[3,1]+G[1,4]*G[3,2]+G[2,3]*G[4,1]+G[1,3]*G[4,2]))
                end
            end
        else
            for (i,ky) in enumerate(linspace(-pi,pi,Grid))
                for (j,kx) in enumerate(linspace(-pi,pi,Grid))
                    if Cum == 0
                        gf_ktilde = build_gf_ktilde_prebuild(model,tk[i,j])
                        G = periodize_nocoex(kx, ky, gf_ktilde)
                    elseif Cum == 1
                        G = stiffness_cum_SC(model, kx, ky)
                    end
                    Sum += 2.0*(kxky[1][i,j]*4.0*(G[1,2]*G[2,1])) ## Factor 2.0 for the spin degrees of freedom
                end
            end
        end
        result_n[n] = (1./(2*Grid^2))*real(Sum) # Factor 1/2 to avoid counting the spin degree of freedom twice
    end
    result_out = hcat(modelvector.wvec_[1:len_sEvec_c],result_n)
    println(result_out)
    return result_out
end

"""
Function used to in-plane dSC superfluid stiffness with periodization. 

#Arguments:

- modelvector: modelvector instance
- kxky::Array{Array{Float64,2},1}: Pre-computed current vertices. 
- tk::Array{Array{Complex{Float64},2},2}: Pre-computed hopping matrix represented in the reduced supercluster BZ.
- Grid::Int64: Value of the k-space grid dimension.
- Cum::Int64: 1 if the periodization is to be done on the cumulant. 0 if the periodization is to be done on Green's function.
- AFMSC::Int64: Integer value set to 1 to compute pure SC superfluid stiffness using the formula in coexisting AF+dSC regime. Otherwize, let it be 0. 
- len_sEvec_c::Int64: Length of the Matsubara frequency grid. Set by default to 500.

#Returns:

- Returns an array of two columns contaning the k-integrated Green's function and the associated Matsubara frequency.

"""
function stiffness_NOCOEX_Per_Cum_ab_k_grid(modelvector::ModelVector, kxky::Array{Array{Float64,2},1}, tk::Array{Array{Complex{Float64},2},2}, Grid::Int64, Cum::Int64, AFMSC::Int64; len_sEvec_c::Int64=500)
    len_sEvec_c > size(modelvector.sEvec_c_)[1] && throw(ErrorException("You have exceeded the number of Matsubara frequencies available for calculations."))
    result_n = Array{Float64,1}(len_sEvec_c)
    for n in 1:len_sEvec_c
        model = Model(modelvector,n)
        Sum = 0.0
        for (i,ky) in enumerate(linspace(-pi,pi,Grid))
            for (j,kx) in enumerate(linspace(-pi,pi,Grid))
                if Cum == 0 && AFMSC == 0
                    gf_ktilde = build_gf_ktilde_prebuild(model,tk[i,j])
                    G = periodize_nocoex(kx, ky, gf_ktilde)
                    Sum += 1.0*(kxky[1][i,j]*4.0*(G[1,2]*G[2,1]))
                elseif Cum == 1 && AFMSC == 0
                    G = periodize_cumulant_SC(model, kx, ky)
                    Sum += 1.0*(kxky[1][i,j]*4.0*(G[1,2]*G[2,1]))
                elseif Cum == 0 && AFMSC == 1 
                    gf_ktilde = build_gf_ktilde_prebuild(model,tk[i,j])
                    G = periodize_AFM_orb(gf_ktilde, kx, ky)
                    Sum += 1.0*(kxky[1][i,j]*4.0*(G[1,3]*G[3,1]+G[2,3]*G[3,2]+G[1,4]*G[4,1]+G[2,4]*G[4,2])+
                    kxky[2][i,j]*2.0*((G[1,3]+G[2,4])*(G[3,2]+G[4,1])+(G[1,4]+G[2,3])*(G[3,1]+G[4,2]))+
                    kxky[2][i,j]*2.0*((G[1,3]+G[2,4])*(G[3,2]+G[4,1])+(G[1,4]+G[2,3])*(G[3,1]+G[4,2]))+
                    kxky[3][i,j]*4.0*(G[2,4]*G[3,1]+G[1,4]*G[3,2]+G[2,3]*G[4,1]+G[1,3]*G[4,2]))
                elseif Cum == 1 && AFMSC == 1
                    G = periodize_coex_cum_finale(model, kx, ky)
                    Sum += 1.0*(kxky[1][i,j]*4.0*(G[1,3]*G[3,1]+G[2,3]*G[3,2]+G[1,4]*G[4,1]+G[2,4]*G[4,2])+
                    kxky[2][i,j]*2.0*((G[1,3]+G[2,4])*(G[3,2]+G[4,1])+(G[1,4]+G[2,3])*(G[3,1]+G[4,2]))+
                    kxky[2][i,j]*2.0*((G[1,3]+G[2,4])*(G[3,2]+G[4,1])+(G[1,4]+G[2,3])*(G[3,1]+G[4,2]))+
                    kxky[3][i,j]*4.0*(G[2,4]*G[3,1]+G[1,4]*G[3,2]+G[2,3]*G[4,1]+G[1,3]*G[4,2]))
                end
            end
        end
        result_n[n] = 1./(Grid^2)*real(Sum) # No factor 2 to count the spin degree of freedom
    end
    result_out = hcat(modelvector.wvec_[1:len_sEvec_c],result_n)
    println(result_out)
    return result_out
end

end ## End of module 
