
function gpu_dtw(query::Array, subject::Array)


    const m=length(query)
    const n=length(subject)
    cost   = Array(Float32, m * m)

	distance = ccall((:gpu_dtw_func, "GPU-DTW"),
              Float32, (Ptr{Float32}, Ptr{Float32}, Int32, Int32, Ptr{Float32},), query, subject, m,n, cost)
    path_len =  ccall((:get_path_length, "GPU-DTW"), Int32, ())

    path_x = Array(Int32,  path_len)
    path_y = Array(Int32,  path_len)

    ccall((:get_path, "GPU-DTW"), Void, (Ptr{Int32}, Ptr{Int32},), path_x, path_y)


    return distance, cost, (path_x, path_y)
   

end


# Main program
f = open("data/query0.bin")
qlens = 1024
query = read(f, Float32,qlens)
close(f)

f = open("data/subject.bin")
slens = 10000
subject = read(f, Float32,slens)
close(f)

distance , cost , path = gpu_dtw(query, subject)


@printf("Distance %f\n", distance);
