using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

function dtw(s,t, use_fast)
    dtw_ = load_op_and_grad("./build/libDtw","dtw", multiple=true)
    s,t, use_fast = convert_to_tensor([s,t,use_fast], [Float64,Float64, Int32])
    dtw_(s,t,use_fast)
end


# TODO: specify your input parameters
Sample = [1,2,3,5,5,5,6]
Test = [1,1,2,2,3,5]

Sample = rand(100)
Test = rand(100)
u, p = dtw(Sample, Test,1)
sess = Session(); init(sess)
@show run(sess, u)


# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m)
    return sum(dtw(Test, m, 1)[1]^2)
end
# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(100))
v_ = rand(100)
y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(v_.*dy_)
end

sess = Session(); init(sess)
sval_ = run(sess, s_)
wval_ = run(sess, w_)
close("all")
loglog(gs_, abs.(sval_), "*-", label="finite difference")
loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
