module Utils

export topological_sort

using DataStructures


# resource: https://github.com/mission-peace/interview/blob/master/src/com/interview/graph/TopologicalSort.java 
function topological_sort(V, A)
    visited = Set()
    sorted = Stack{String}()
    E = [(from, to) for from in keys(A) for to in A[from]]  # edges
    for v in V
        if v in visited
            continue
        end
        visited, sorted = top_sort(v, E, visited, sorted)
    end
    return [pop!(sorted) for _ in 1:length(sorted)]
end


function top_sort(v, E, visited, sorted)
    push!(visited, v)
    for c in get_children(v, E)
        if c in visited
            continue
        end
        visited, sorted = top_sort(c, E, visited, sorted)
    end
    push!(sorted, v)
    return visited, sorted
end


function get_children(v, E)
    children = []
    for e in E
        if e[1] == v
            push!(children, e[2])
        end
    end
    return children
end

end
