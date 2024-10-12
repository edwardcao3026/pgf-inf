using LinearAlgebra, StatsBase, Distributions, SparseArrays, FastGaussQuadrature, HypergeometricFunctions, Optim
using DelimitedFiles, Random, StatsPlots, Statistics
using DataFrames
using HypothesisTests
using Distributed
using Plots
using StatsBase, Plots, CSV,DataFrames,DelimitedFiles
using FastGaussQuadrature, HypergeometricFunctions, Optim
using KernelDensity, Interpolations
u_counts = Float64.(readdlm("synthetic_data.csv",',')[2:end,1])
s_counts = Float64.(readdlm("synthetic_data.csv",',')[2:end,2])
β = Float64.(readdlm("synthetic_data.csv",',')[2:end,3])
# β Gauss Quadrature
Density = kde(β)
interp = LinearInterpolation(Density.x, Density.density, extrapolation_bc=Interpolations.Flat())
min_β = minimum(β)
max_β = maximum(β)
xl, wl = gausslegendre(7)
Xl = (max_β-min_β)/2*xl.+(max_β+min_β)/2
pf = interp.(Xl)
sum(wl.*pf*(max_β-min_β)/2)
# Gauss Quadrature
x, w = gausslegendre(7)
x = (x.+1)./2
X = [(x[i],x[j]) for i = 1 :length(x) for j = 1:length(x)]
W = vec(w*w')

function cus_hist2(data1::Vector,data2::Vector)
    data = (data1,data2)
    max1 = maximum(data1)
    max2 = maximum(data2)
    edge1 = collect(0:1:max1+1)
    edge2 = collect(0:1:max2+1)
    h = fit(Histogram, data,(edge1.-0.5,edge2.-0.5))
    Weights = h.weights/length(data1)
    return Weights
end
function model_gf2(ps,β,z1,z2,sel)
    if sel == 1
        ℓ,τ = ps
        ρ=ℓ*β
        G_p=exp(ρ*(z2-1))*exp(ρ*τ*(z1-1))
        return (G_p)^2
    elseif sel == 2
        ℓ,σon,σoff,τ = ps
        ρ=ℓ*β
        dm=1.
        u1=z1-1
        u2=z2-1
        x1=ρ*u1/dm
        x2=ρ*u2/dm
        α=σon/(σoff+σon)
        r=1+(σoff+σon)/dm-x1
        θ=sqrt(Complex(((σoff+σon)/dm-x1)^2+4*σon*x1/dm))
        v1=dm*(r+θ-1)/2
        v2=dm*(r-θ-1)/2
        G=(v1*exp(-v2*τ)-v2*exp(-v1*τ))/(dm*θ)*pFq((σon/dm, ), ((σon+σoff)/dm, ), x2)+ρ*u1*(exp(-v2*τ)-exp(-v1*τ))/(dm*θ)*σon/(σoff+σon)*pFq((1+σon/dm, ), (1+(σoff+σon)/dm, ), x2)
        return real(G^2)
    end
end
function model_tgf(ps,z1,z2,sel)
    mtgf = [model_gf2(ps,Xl[i],z1,z2,sel)*pf[i]*wl[i] for i = 1 : length(Xl)]
    return sum(mtgf)*(max_β-min_β)/2
end
function hist_gf2(hist_data,z1,z2)
    Nx = size(hist_data,1)
    Ny = size(hist_data,2)
    z1_vec = [z1.^i for i = 0 : Nx-1]
    z2_vec = [z2.^i for i = 0 : Ny-1]
    z_mat = z1_vec*z2_vec'
    return sum(z_mat.*hist_data)
end


# Distance function
function sdist2(hist_data,ps,z,a,sel)
    z1,z2=z
    mtgf = model_tgf(ps,z1,z2,sel)
    return mtgf^(1+a) - mtgf^a * hist_gf2(hist_data,z1,z2) * (1+1/a) + hist_gf2(hist_data,z1,z2)^2
end

# Objective function
function int_dist2(ps,hist_data,a,X,W,sel)
    N = length(W)
    dist = zeros(N)
    for i = 1 : N
        dist[i] = sdist2(hist_data,ps,X[i],a,sel)
    end
    return sum(W.*dist)/4
end

function inf_err2(D1::Vector,D2::Vector,sel)
    hist_data = cus_hist2(D1,D2)
    if sel == 1
        init_ps = [0.0;0.0]
        results = optimize(ps->int_dist2(exp.(ps),hist_data,1,X,W,sel),init_ps,Optim.Options(show_trace=false,g_tol=1e-20,iterations = 1000)).minimizer
    elseif sel == 2
        init_ps = [0.0;0.0;0.0;0.0]
        results = optimize(ps->int_dist2(exp.(ps),hist_data,1,X,W,sel),init_ps,Optim.Options(show_trace=false,g_tol=1e-20,iterations = 1000)).minimizer
    end 
    return exp.(results)
end




function model_select(err)
        aerr = mean(err,dims=1)'
        best_aerr,ind = findmin(aerr)
        best_std = std(err[:,ind[1]])
        tr = best_aerr .+ [best_std*sqrt(1-(cor(err[:,i],err[:,ind[1]]))) for i = 1 :2]
        best_model = 0
        flg = vec(Float64.(aerr .< tr))
        if flg == zeros(2)
            best_model = ind[1]
        else
            best_model,~ =  findmin(vcat(collect(1:2)[flg[:,1] .== 1],ind[1]))
        end
        return best_model
end

function err_cross(u_counts,s_counts)
    chunk_size = round(Int,length(u_counts)/10)
    new_arrays_n=Vector[]
    new_arrays_n = [u_counts[j:min(j+chunk_size-1, end)] for j in 1:chunk_size:chunk_size*9]
    new_arrays_n = push!(new_arrays_n, u_counts[chunk_size*9+1:end])
    new_arrays_m=Vector[]
    new_arrays_m = [s_counts[j:min(j+chunk_size-1, end)] for j in 1:chunk_size:chunk_size*9]
    new_arrays_m = push!(new_arrays_m, s_counts[chunk_size*9+1:end])
    err = zeros(10,2)
    ps = Vector{Vector{Float64}}()
    for sel = 1 : 2
        for i = 1 : 10
            temp = collect(1:10)
            sig = .!(temp.==i)
            rdb = vec(vcat(new_arrays_m[sig]...))
            rda = vec(vcat(new_arrays_n[sig]...))
            push!(ps,vec(inf_err2(rda,rdb,sel)))
            err[i,sel] = int_dist2(ps[i+(sel-1)*10],cus_hist2(new_arrays_n[i],new_arrays_m[i]),1,X,W,sel)
        end
    end
    #writedlm("merfish_cross_val_1/2_renew_ps_$(index).csv",ps,',')
    #writedlm("merfish_cross_val_1/2_renew_err_$(index).csv",err,',')
    model_select(err)
    return model_select(err)
end

err_cross(u_counts,s_counts)