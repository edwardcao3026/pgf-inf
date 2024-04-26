using Plots, CSV,DataFrames
# data loading
data = CSV.File("ssa.csv") |> DataFrame
u_counts = data.Nuclear_mRNA
s_counts = data.Cytoplasmic_mRNA
β = data.Cell_volume

include("PGF_function.jl")
include("FSP_function.jl")
gr()

init_ps = [1.0;1.0;1.0;1.0]
hist_data = cus_hist2(u_counts,s_counts)
@time results_PGF = optimize(ps->int_dist(exp.(ps),hist_data,X,W),init_ps,Optim.Options(show_trace=true,g_tol=1e-8,iterations = 500)).minimizer
ps_PGF=exp.(results_PGF)

@time results_FSP = optimize(ps->sum_pro_delay(exp.(ps),u_counts,s_counts),init_ps,Optim.Options(show_trace=true,g_tol=1e-8,iterations = 500)).minimizer
ps_FSP=exp.(results_FSP)

joint_PGF = conv_function_delay(ps_PGF)
mar_u_PGF = vec(sum(joint_PGF,dims = 2))
mar_s_PGF = vec(reshape(sum(joint_PGF,dims = 1),(89,1)))

joint_FSP = conv_function_delay(ps_FSP)
mar_u_FSP = vec(sum(joint_PGF,dims = 2))
mar_s_FSP = vec(reshape(sum(joint_PGF,dims = 1),(89,1)))

x=[0:maximum(u_counts)]
plot(cus_hist(u_counts),label="smFISH",title="Nuclear_mRNA")

plot!(mar_u_PGF,label="PGF")

plot!(mar_u_FSP,label="FSP")

x=[0:maximum(s_counts)]
plot(cus_hist(s_counts),label="smFISH",title="Cytoplasmic_mRNA")

plot!(mar_s_PGF,label="PGF")

plot!(mar_s_FSP,label="FSP")
