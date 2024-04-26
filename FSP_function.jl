using SparseArrays, StatsBase, LinearAlgebra
using DSP
using Distributions, SparseArrays, FastGaussQuadrature, Optim
using DifferentialEquations
using KernelDensity, Interpolations

# β Gauss Quadrature
Density = kde(β)
interp = LinearInterpolation(Density.x, Density.density, extrapolation_bc=Interpolations.Flat())
min_β = minimum(β)
max_β = maximum(β)
xl, wl = gausslegendre(7)
Xl = (max_β-min_β)/2*xl.+(max_β+min_β)/2
pf = interp.(Xl)

function CME_aux!(du,u,p,t)
    ρ,σon,σoff,N=p
    # Define transition matrix for \bar{P}'s CME
    B = zeros(2*N,2*N)
    B[1:N,1:N] = - spdiagm(0 => σon*ones(N)) 
    B[1:N,N+1:2*N] = spdiagm(0 => σoff*ones(N)) 
    B[N+1:2*N,1:N] = spdiagm(0 => σon*ones(N)) 
    B[N+1:2*N,N+1:2*N] = - spdiagm(0 => σoff*ones(N))  - spdiagm(0 => ρ*vcat(ones(N-1),0)) + spdiagm(-1 => ρ*ones(N-1))
    du[1:end] = B*u

    return u
end

function CME_maturemar!(p)
    ρ,σon,σoff,d,N=p
    # Define transition matrix for \bar{P}'s CME
    C = zeros(2*N,2*N)
    C[1:N,1:N] = - spdiagm(0 => σon*ones(N)) - spdiagm(0 => d*collect(0:N-1)) + spdiagm(1 => d*collect(1:N-1))
    C[1:N,N+1:2*N] = spdiagm(0 => σoff*ones(N)) 
    C[N+1:2*N,1:N] = spdiagm(0 => σon*ones(N)) 
    C[N+1:2*N,N+1:2*N] = - spdiagm(0 => σoff*ones(N))  - spdiagm(0 => ρ*vcat(ones(N-1),0)) + spdiagm(-1 => ρ*ones(N-1))- spdiagm(0 => d*collect(0:N-1)) + spdiagm(1 => d*collect(1:N-1))
    C[1,:].=1 
    pp=C\[1;zeros(2*N-1)]
    pm1=pp[N+1:2*N]

    return pm1
end

function CME_main!(p1)
    ρ,σon,σoff,d,N1,N2,pn0,pn1,pm1=p1
        # Define transition matrix for Q's CME
        M=zeros(2*N1*N2,2*N1*N2)
        for i =1:N2 
            M[(i-1)*N1+1:i*N1 ,(i-1)*N1+1:i*N1]=-spdiagm(0=> σon*ones(N1))-spdiagm(d*(i-1)*ones(N1))
            M[(i-1)*N1+1:i*N1 ,(i-1)*N1+1+N1*N2:i*N1+N1*N2]=spdiagm(0=>σoff*ones(N1))
            M[(i-1)*N1+1+N1*N2:i*N1+N1*N2 ,(i-1)*N1+1:i*N1]=spdiagm(0=>σon*ones(N1))
            M[(i-1)*N1+1+N1*N2:i*N1+N1*N2 ,(i-1)*N1+1+N1*N2:i*N1+N1*N2]=-spdiagm(0=>σoff*ones(N1))-spdiagm(d*(i-1)*ones(N1))-spdiagm(0=>ρ*vcat(ones(N1-1),0))+spdiagm(-1=>ρ*ones(N1-1))
        end
        
        for i =1:N2-1 
            M[(i-1)*N1+1:i*N1 , i*N1+1:(i+1)*N1]=spdiagm(d*i*ones(N1))
            M[(i-1)*N1+1+N1*N2:i*N1+N1*N2 , i*N1+1+N1*N2:(i+1)*N1+N1*N2]=spdiagm(d*i*ones(N1))
        end

    D=-ρ * vcat(reshape(pn0*vcat(0,pm1[1:end-1])'-vcat(0,pn0[1:end-1])*pm1',(N1*N2,1)),reshape(pn1*vcat(0,pm1[1:end-1])'-vcat(0,pn1[1:end-1])*pm1',(N1*N2,1))).+[1;zeros(2*N1*N2-1)]
    M[1,:].=1 
    solution=M\D

    p_0 = solution[1:N1*N2]
    p_1 = solution[N1*N2+1:2*N1*N2]
    u_intk = reshape(p_0+p_1,(N1,N2))

    U_intk = max.(0.0,u_intk)
    U_intk = U_intk/sum(U_intk)

    return U_intk
end

function delaysolG1(pt,β,N1,N2)
    ℓ,σoff,σon,τ = pt
    ρ = β*ℓ
    d = 1
    # Define transition matrix without delay effect terms

    # Obtain the edge probability of Nuclear mRNA at τ 
    pn = (ρ,σon,σoff,N1)
    u_aux = zeros(2*N1)
    # Initial condition -- gene state ON (always true)
    u_aux[N1+1] = 1.
    tspan = (0.0, τ)
    prob2 = ODEProblem(CME_aux!, u_aux, tspan, pn)
    sol2 = solve(prob2, Tsit5(), saveat=0.2)
    pn0 = sol2.u[end][1:N1]
    pn1 = sol2.u[end][N1+1:end]
  
    # Obtain the edge probability of Cytoplasmic mRNA at steady state
    pm = (ρ,σon,σoff,d,N2)
    pm1=CME_maturemar!(pm)

    # Obtain joint probability distribution
    p1=(ρ,σon,σoff,d,N1,N2,pn0,pn1,pm1)
    u_int=CME_main!(p1)
    return u_int
    
end

function conv_function_delay(pt)
    N1 = 45
    N2 = 45
    pro = zeros(2*N1-1,2*N2-1)
    for i in 1:length(Xl) 
        ll = delaysolG1(pt,Xl[i],N1,N2)
        LL = ll
        LL2 = conv(LL,LL)*wl[i]*pf[i]
        pro = pro+LL2
    end
    
    return pro
end

function sum_pro_delay(pt,u_counts,s_counts)
    prob= conv_function_delay(pt)
    sum_prob=0
    for j in 1:length(u_counts)
        a=log(prob[u_counts[j]+1,s_counts[j]+1]+1e-10)
        sum_prob=a+sum_prob
    end
    return -sum_prob
end

function cus_hist(data::Vector)
    max = maximum(data)
    edge = collect(0:1:max+1)
    h = fit(Histogram, data,edge.-0.5)
    Weights = h.weights/length(data)
    return Weights
end