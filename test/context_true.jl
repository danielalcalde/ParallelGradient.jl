l1 = 5
l2 = 6
l3 = 7
l4 = 1

@everywhere function func(d, x)
    return d(x)
end

@everywhere function func_scalar(d, x)
    return (sum(d(x)), 0)
end


d1 = Dense(l1=>l2, tanh)
d2 = Dense(l2=>l3, tanh)
d3 = Dense(l3=>l4, tanh)
d = Chain(d1, d2, d3)

c = randn(Float32, l1, 9)
y = randn(Float32, l4, 9)

loss_simple(d) = sum(abs2.(d(c) .- y))

function loss(d, map_func)
    o = d[1](c)
    o = map_func(ci->func(d[2], ci), eachslice(o, dims=2))
    o = ParallelGradient.hcat_vec(o)
    o = d[3](o)
    return sum(abs2.(o.- y))
end

losses = OrderedDict(
    "loss" => loss,
    )


maps = OrderedDict(
    "ptmap" => ptmap,
    "dpmap" => dpmap,
    "dpmap_chuncked" => (args...; kwargs...) -> dpmap(args...; pmap_function=pmap_chuncked, kwargs...),
    "dtmap" => dtmap,
    )

l_true = Dict()
g_true = Dict()
for (name_l, loss_i) in losses
    l_true[name_l], g_true[name_l] = withgradient(Flux.params(d)) do
        loss(d, map)
    end
end

for (name, map_i) in maps, (name_l, loss_i) in losses
    @testset "$(name)_$(name_l)" begin
            @test loss(d, map_i) ≈ l_true[name_l]

            l, g = withgradient(Flux.params(d)) do
                loss_i(d, map_i)
            end
            @test l ≈ l_true[name_l]
            @test g ≈ g_true[name_l]
    end
end


function loss_scalar(d, map_func)
    o = map_func(ci->func_scalar(d, ci)[1], eachslice(c, dims=2))
    return sum(abs2.(o .- y[1,:]))
end

losses_scalar = OrderedDict(
    "loss_scalar" => loss_scalar,
    )


maps_scalar = OrderedDict(
    "dpmap" => dpmap_scalar,
    )

for (name_l, loss_i) in losses_scalar
    l_true[name_l], g_true[name_l] = withgradient(Flux.params(d)) do
        loss(d, map)
    end
end

for (name, map_i) in maps_scalar, (name_l, loss_i) in losses_scalar
    @testset "$(name)_$(name_l)" begin
            @test loss(d, map_i) ≈ l_true[name_l]

            l, g = withgradient(Flux.params(d)) do
                loss_i(d, map_i)
            end
            @test l ≈ l_true[name_l]
            @test g ≈ g_true[name_l]
    end
end