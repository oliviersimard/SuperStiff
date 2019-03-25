
d=2;
A=Array{Int64,3}(d,d,d); B=Array{Int64,3}(d,d,d); C=zeros(Int64,d,d)
for i in 1:d,j in 1:d,k in 1:d
    A[i,j,k] = 2i+j^2-k^3
    B[i,j,k] = i+2j+4k
end

tic()
for i in 1:size(A,1)
    for j in 1:size(A,3)
        for k in 1:size(A,1)
            for n in 1:size(B,3)
                C[j,n] += A[i,j,k]*B[k,i,n]
            end
        end
    end
end
toc()
print("Matrix C is ", C)

A_permuted = permutedims(A,[2,1,3]); B_permuted = permutedims(B,[2,1,3]);
A_reshaped = reshape(A_permuted,2,4); B_reshaped = reshape(B_permuted,4,2);
tic()
C_reshaped = A_reshaped*B_reshaped;
toc()
print("Matrix C is ", C_reshaped)

A=rand(Int64,100,100,100); B=rand(Int64,100,100,100); 
C=zeros(Int64,100,100);

tic()
for i in 1:size(A,1)
    for j in 1:size(A,3)
        for k in 1:size(A,1)
            for n in 1:size(B,3)
                C[j,n] += A[i,j,k]*B[k,i,n]
            end
        end
    end
end
toc()

A_reshaped=reshape(permutedims(A,[2,1,3]),100,10000); 
B_reshaped=reshape(permutedims(B,[2,1,3]),10000,100);
tic()
C_reshaped = A_reshaped*B_reshaped;
toc()

X=Complex{Float64}[0.0 1.0;1.0 0.0]; 
Y=Complex{Float64}[0.0im -1.0im;1.0im 0.0im]; 
Z=Complex{Float64}[1.0 0.0;0.0 -1.0];
h=kron(X,X)+kron(Y,Y)+kron(Z,Z);
N=4; C=2; d=2;
H=zeros(Complex{Float64},d^N,d^N);

for l in 0:N-C
    H+=kron(kron(eye(d^l),h),eye(d^(N-C-l)))
end
E,V = eig(H)
print("The first eigenvalues are: ", E[1:11])

function diagonalization_model(h::Matrix{T},N::Int64;C::Int64=2,
        d::Int64=2,method_sparse::Bool=true) where {T}
    H=zeros(T,d^N,d^N)
    for l in 0:N-C
        H+=kron(kron(eye(d^l),h),eye(d^(N-C-l),d^(N-C-l)))
    end
    if method_sparse
        E,V = eigs(H;nev=10,which=:SR)
    else
        E,V = eig(H)
    end
    return E
end

@time diagonalization_model(h,12)

@time diagonalization_model(h,12,method_sparse=false)

function permR2(i::Int64,P::Int64)
    return mod.(collect(1:P)+i-2,P)+1
end

function permL2(i::Int64,P::Int64)
    return mod.(collect(1:P)-i,P)+1
end

function H_wavefunction(h_tilde::Matrix{T},phi::Array{T,P},N::Int64;
        OBC::Int64=1,d::Int64=2,C::Int64=2) where {T,P}
    H_phi = zeros(T,fill(d,C+1)...,d^(N-C-1))
    for i in 1:N-OBC
        if N == 1
            phi = permutedims(phi,permR2(1,P))
            H_phi = permutedims(H_phi,permR2(1,P))
            phi_tmp = h_tilde*reshape(phi,d^C,d^(N-C))
            phi_tmp = reshape(phi_tmp,fill(d,C+1)...,d^(N-C-1))
            H_phi = reshape(H_phi,fill(d,C+1)...,d^(N-C-1))
            H_phi += phi_tmp
        else
            phi = permutedims(phi,permR2(2,P))
            H_phi = permutedims(H_phi,permR2(2,P))
            phi_tmp = h_tilde*reshape(phi,d^C,d^(N-C))
            phi_tmp = reshape(phi_tmp,fill(d,C+1)...,d^(N-C-1))
            H_phi = reshape(H_phi,fill(d,C+1)...,d^(N-C-1))
            phi = reshape(phi,fill(d,C+1)...,d^(N-C-1))
            H_phi += phi_tmp
        end
    end
    if OBC == 1
        return(reshape(permutedims(H_phi,permR2(2,P)),d^N))
    elseif OBC == 0
        return(reshape(H_phi,d^N))
    end
end

function power_method(h::Matrix{T},N::Int64;OBC::Int64=1,d::Int64=2,
        C::Int64=2,tol::Float64=1e-4) where {T}
    #phi = rand(T,fill(d,C+1)...,d^(N-C-1))
    #phi = collect(Complex{Float64},1:2^N)
    phi = rand(T,d^N)
    phi = phi/sqrt(dot(conj(phi),phi))
    phi = reshape(phi,fill(d,C+1)...,d^(N-C-1))
    list_E_tmp = []
    E,V = eig(h)
    alpha=max(E...)
    h_tilde = h - alpha*kron(eye(d),eye(d))
    i = 1
    bool = true
    while bool
        H_phi = H_wavefunction(h_tilde,phi,N,OBC=OBC)
        #H_phi = H_phi/dot(conj(H_phi),H_phi)
        phi = reshape(phi,d^N)
        #E = dot(conj(phi),H_phi)/dot(conj(phi),phi) + α*(N-OBC)
        E = dot(conj(phi),H_phi) + alpha*(N-OBC)
        #println("E = ", E)
        push!(list_E_tmp,E)
        if i>2 && abs(list_E_tmp[i]-list_E_tmp[i-1])<tol
           bool = false
           println("Desired precision reached!")
        end
        i+=1
        phi = reshape(phi,fill(d,C+1)...,d^(N-C-1))
        phi = reshape(H_phi/sqrt(dot(conj(H_phi),H_phi)),fill(d,C+1)...,d^(N-C-1))
    end
    println("E = ",E)
    println("Number of iterations = ", i)
end

@time power_method(h,4,tol=1e-9)

@time power_method(h,12,tol=1e-9)

@time power_method(h,16,OBC=0,tol=1e-9)

@time power_method(h,16,tol=1e-9)

@time power_method(h,25)

@time power_method(h,25,OBC=0)
