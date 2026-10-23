root = dirname(@__DIR__)
count = 0
for directory in ("simulations", "vendor")
    for (folder, _, files) in walkdir(joinpath(root, directory))
        for file in files
            endswith(file, ".jl") || continue
            path = joinpath(folder, file)
            ast = Meta.parseall(read(path, String); filename=path)
            function check_parse(node)
                node isa Expr || return
                node.head in (:error, :incomplete) && error("Parse error in $path: $node")
                foreach(check_parse, node.args)
            end
            check_parse(ast)
            global count += 1
        end
    end
end
println("Parsed $count Julia source files successfully.")
