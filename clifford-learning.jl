
include("random_clifford/Initial.jl")
include("random_clifford/Symplectic.jl")

using Yao
using YaoBlocks
using Random
using PyPlot
using YaoSym
using YaoExtensions
using QuAlgorithmZoo: Adam, update!
using LinearAlgebra

n_qubits = 5
n_circuits = 100
n = n_qubits;

n_clifford_circuits = max(getNumberOfSymplecticCliffords(n), 2^63-1)

function parseCommand(command)
    if command[1] == "hadamard"
        return put(command[2]=>H)
    elseif command[1] == "phase"
        return put(command[2]=>shift(π/2))
    elseif command[1] == "cnot"
        return control(command[2][1], command[2][2]=>X)
    else
        error("The command should be either 'hadamard', 'phase' or 'cnot', not $(command[1])")
    end
end

function getRandomCircuit(i)
    commands = decompose(rand(0:n_clifford_circuits), rand(0:2^(2*n)), n, true)
    list_commands = map(c -> (c[1], parse.(Int, split(c[2], ","))),
                    split.(commands, r"\(|\)", keepempty=false)[2:length(commands)-1])
    circuit = chain(n_qubits, parseCommand.(list_commands))
    print("$(i) ")
    return circuit
end

list_circuits = [getRandomCircuit(i) for i in 1:n_circuits];

W = 1/n * (UnitUpperTriangular(ones(n, n)) - I)

function get_hamiltonian(W::AbstractMatrix)
    nbit = size(W, 1)
    ab = Any[]
    for i=1:nbit,j=i+1:nbit
        if W[i,j] != 0
            push!(ab, 0.5*W[i,j]*repeat(nbit, Z, [i,j]))
        end
    end
    sum(ab)
end

hamil = get_hamiltonian(W);

function evaluate_hamiltonian(i)
    print(i);
    print(" ")
    expect(hamil, zero_state(n_qubits) |> list_circuits[i])
end

state = zero_state(n_qubits) |> list_circuits[1];

energies = evaluate_hamiltonian.(1:n_circuits);

hist(energies, normed=true);

v_unitary = variational_circuit(n);
v_circuit = chain(n+1, 1=>H, control(1, 2:n+1=>v_unitary), 1=>H);
total_circuits = [chain(n+1, concentrate(n+1, sp, 2:n+1), v_circuit) for sp in list_circuits];






function loss(real_energy, predicted_energy)
    return 1/2 * (real_energy - predicted_energy)^2
end

optimizer = Adam(lr=0.01)
params = parameters(v_unitary)
niter = 100

for i = 1:niter
    ## `expect'` gives the gradient of an observable.
    grad_input, grad_params = expect'(put(n+1, 1=>Z), zero_state(n+1) => total_circuits[i])
    grad_input *= (1=>Z, zero_state(n+1) => total_circuits[i]) - energies[i]

    ## feed the gradients into the circuit.
    dispatch!(c, update!(params, grad_params, optimizer))
    println("Step $i, Energy = $(expect(hami, zero_state(N) |> c))")
    "$(expect(1=>Z, zero_state(n+1) => total_circuits[i]))"
end

YaoBlocks.expect'

total_circuits[1]|>typeof

methods(expect')


