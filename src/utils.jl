function gpu(backend::Backend, x::Array)
    y = allocate(backend, eltype(x), size(x))
    return copyto!(y, x)
end
