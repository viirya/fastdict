
arg <- commandArgs(trailingOnly = TRUE)
argLen <- length(arg)
if (argLen == 3) {
    arg_b <- as.integer(arg[1]) # total b bits
    arg_n <- as.integer(arg[2]) # total n binary codes
    arg_t <- as.double(arg[3]) # number of bits for bit count


    first_ele = c(1:min(arg_b, ceiling(log2(arg_n)) + 1))
    second_ele = c(min(arg_b, ceiling(log2(arg_n)) + 1):arg_b)

    print(first_ele)
    print(second_ele)

    worst_weight = 0.03
    best_weight = 0.97

    first_sum = sum(2 ^ first_ele) * worst_weight + sum(2 * length(first_ele)) * best_weight

    second_sum = sum(rep(arg_n + 1, (arg_b - log2(arg_n)))) * worst_weight + sum(2 * length(second_ele)) * best_weight

    print("result:")
    compressed_bits = (first_sum + second_sum) * arg_t
    original_bits = as.double(arg_b) * arg_n

    print(paste("original: ", original_bits, sep = ''))
    print(paste("compressed: ", compressed_bits, sep = ''))
   
    if (compressed_bits < original_bits) {
        print("compression smaller than original")
        diff = original_bits - compressed_bits
        print(paste("when compressing ", original_bits / 8 / 1024 / 1024, " Mbytes codes", sep = ''))
        print(paste("about: ", diff / 8 / 1024 / 1024, " Mbytes", sep = ''))
    } else {
        print("compression bigger than original")
    }
}

