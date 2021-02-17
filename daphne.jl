module Daphne

export daphne

using JSON


function daphne(args, cwd = "../daphne")
    output = cd(() -> read(`lein run -f json $args`, String), cwd)
    return JSON.parse(output)
end


end

# args = ["desugar", "-i", "../HW2/programs/tests/deterministic/test_1.daphne"]
# lein run -f json desugar -i ../HW2/programs/tests/deterministic/test_1.daphne
