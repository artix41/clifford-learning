include("random_clifford/Initial.jl")
include("random_clifford/Symplectic.jl")

using ArgParse
using Yao
using YaoBlocks
using LinearAlgebra
using PyPlot

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "n_qubits"
            help = "Number of qubits"
            arg_type = Int
            default = 2
        "n_circuits"
            help = "Number of circuits to generate (size of the dataset)"
            arg_type = Int
            default = 100
    end

    return parse_args(s)
end

function parse_command(command)
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

function get_random_commands(i)
    commands = decompose(rand(0:n_clifford_circuits), rand(0:2^(2*n)), n, true)[2:end-1]
    print("$i ")
    return commands
end

function get_circuit(commands)
    commands = map(c -> (c[1], parse.(Int, split(c[2], ","))),
                   split.(commands, r"\(|\)", keepempty=false))
    circuit = chain(n_qubits, parse_command.(commands))
    return circuit
end

function get_hamiltonian(W::AbstractMatrix)
    nbit = size(W, 1)
    ab = Any[]
    for i=1:nbit,j=i+1:nbit
        if W[i,j] != 0
            push!(ab, W[i,j]*repeat(nbit, Z, [i,j]))
        end
    end
    sum(ab)
end

function evaluate_hamiltonian(i)
    return expect(hamil, zero_state(n_qubits) |> list_circuits[i])
end

parsed_args = parse_commandline()

n_qubits = parsed_args["n_qubits"]
n_circuits = parsed_args["n_circuits"]
n = n_qubits

n_clifford_circuits = max(getNumberOfSymplecticCliffords(n), 2^63-1)

list_commands = [get_random_commands(i) for i in 1:n_circuits]
list_circuits = get_circuit.(list_commands)

J = 1/n * (UnitUpperTriangular(ones(n, n)) - I)

hamil = get_hamiltonian(J)

energies = evaluate_hamiltonian.(1:n_circuits);
hist(energies)

output_folder = "data/$(n_qubits)q/"

if !ispath(output_folder)
    mkdir(output_folder)
end

open(joinpath(output_folder, "circuits.txt"), "w") do f
    for circuit in list_commands
        for command in circuit
            write(f, "$command ")
        end
        write(f, "\n")
    end
end

open(joinpath(output_folder, "energies.txt"), "w") do f
    for energy in energies
        write(f, "$(real(energy))\n")
    end
end