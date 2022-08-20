export get_query_stat, print_query_stat, reset_query_stat

function get_query_stat(p::Potential)
    return [p.query_u, p.query_gradu, p.query_hessu, p.query_laplaceu]
end

function print_query_stat(p::Potential)
    a, b, c, d = get_query_stat(p)
    @printf("query (U): %12d\n", a)
    @printf("query (∇U): %11d\n", b)
    @printf("query (∇²U): %10d\n", c)
    @printf("query (ΔU): %11d\n", d)
end

function reset_query_stat(p::Potential)
    p.query_u = UInt128(0)
    p.query_gradu = UInt128(0)
    p.query_hessu = UInt128(0)
    p.query_laplaceu = UInt128(0)
    return nothing
end
