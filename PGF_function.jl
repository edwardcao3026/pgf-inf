using StatsBase, Plots, CSV,DataFrames
using FastGaussQuadrature, HypergeometricFunctions, Optim
using KernelDensity, Interpolations


# β Gauss Quadrature
Density = kde(β)
interp = LinearInterpolation(Density.x, Density.density, extrapolation_bc=Interpolations.Flat())
min_β = minimum(β)
max_β = maximum(β)
xl, wl = gausslegendre(7)
Xl = (max_β-min_β)/2*xl.+(max_β+min_β)/2
pf = interp.(Xl)

# Gauss Quadrature
x, w = gausslegendre(7)
x = (x.+1)./2
X = [(x[i],x[j]) for i = 1 :length(x) for j = 1:length(x)]
W = vec(w*w')

# Histogram for two dimensions
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

# Generating function of the solution
function model_gf(ps,β,z1,z2)
    ℓ,σoff,σon,τ = ps
    ρ = ℓ*β
    dm=1
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

function model_tgf(ps,z1,z2)
    mtgf = [model_gf(ps,Xl[i],z1,z2)*pf[i]*wl[i] for i = 1 : length(Xl)]
    return sum(mtgf)*(max_β-min_β)/2
end

# Empirical generating function obtained from histogram
function hist_gf(hist_data,z1,z2)
    Nx = size(hist_data,1)
    Ny = size(hist_data,2)
    z1_vec = [z1.^i for i = 0 : Nx-1]
    z2_vec = [z2.^i for i = 0 : Ny-1]
    z_mat = z1_vec*z2_vec'
    return sum(z_mat.*hist_data)
end

# Distance function
function sdist(hist_data,ps,z)
    z1,z2 = z
    a=1
    mtgf = model_tgf(ps,z1,z2)
    return mtgf^(1+a) - mtgf^a * hist_gf(hist_data,z1,z2) * (1+1/a) + hist_gf(hist_data,z1,z2) / a
end

# Objective function
function int_dist(ps,hist_data,X,W)
    N = length(W)
    dist = zeros(N)
    for i = 1 : N
        dist[i] = sdist(hist_data,ps,X[i])
    end
    return sum(W.*dist)/4
end