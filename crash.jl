using BSON
using Flux
using CUDA
using Statistics

struct Critic{T, input_dim, M, N, activation_function}
  model
end
function Critic{T, input_dim, M, N, activation_function}() where {T<:AbstractFloat, input_dim, M, N, activation_function}
  model = Chain(
    Dense(input_dim, M, activation_function),
    Dense(M, N, activation_function),
    Dense(N, 1))
  Critic{T, input_dim, M, N, activation_function}(model)
end
(c::Critic)(x) = c.model(x)
function Flux.functor(::Type{<:Critic{T, input_dim, M, N, activation_function}}, c)  where {T<:AbstractFloat, input_dim, M, N, activation_function}
  ((c.model,), c -> Critic{T, input_dim, M, N, activation_function}(c...))
end

stuff = BSON.load("crash.bson")

function test(c1, c2, i1, i2, q_target, factor)
    critic_params = Flux.params(c1, c2)
    critic_loss = Float32(0)

    c_grad = Flux.gradient(critic_params) do
        critic_loss = mean((c1(i1)[1, :] .- q_target).^2 .+ (c2(i1)[1, :] .- q_target).^2) * Float32(factor)
        critic_loss
    end
    if any([any(isnan.(c_grad[p])) for p in critic_params])
        println("\tq_target $(any(isnan.(q_target)))")
        println("\ti1 $(any(isnan.(i1)))")
        println("\ti2 $(any(isnan.(i2)))")
        println("\tc1(i1) $(any(isnan.(c1(i1))))")
        println("\tc2(i1) $(any(isnan.(c2(i1))))")
        println("\tGradient contains NaN")
        println("\tTest failed (factor $factor)")
    end
    c_grad
end

c1 = stuff[:c1]
c2 = stuff[:c2]
i1 = stuff[:i1]
i2 = stuff[:i2]
q_target = stuff[:q_target]

for c in [c1, c2]
  if any([any(isnan.(p)) for p in Flux.params(c)])
    throw("NaN found")
  end
end
for a in [i1, i2, q_target]
  if any(isnan.(a))
    throw("NaN found")
  end
end

println("Testing CPU")
[test(c1, c2, i1, i2, q_target, f) for f in 2.0 .^ (collect(1:300) .- 150)]

c1_gpu = c1 |> gpu
c2_gpu = c2 |> gpu
i1_gpu = i1 |> gpu
i2_gpu = i2 |> gpu
q_target_gpu = q_target |> gpu
println("Testing GPU: 1.0")
test(c2_gpu, c2_gpu, i1_gpu, i2_gpu, q_target_gpu, 1.0)
println("Testing GPU: 0.5")
test(c2_gpu, c2_gpu, i1_gpu, i2_gpu, q_target_gpu, 0.5)