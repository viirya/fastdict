
draw_figure_ggplot <- function(dataname, data_frames) {

    library(ggplot2)
 
    print(data_frames)

    p <- ggplot(data_frames, aes(x = probe, y = nn, group = distance))
    p + geom_line(aes(colour = distance)) + scale_colour_discrete(h = c(0, 360) + 15, c = 100, h.start = 0, direction = 1) + xlab("Number of multi-probe") + ylab("NN percentage in theory") + theme_bw() + theme(axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15), legend.text = element_text(size = 15), legend.title = element_text(size = 15), axis.text = element_text(size = 15))

    ggsave(file = paste(dataname, ".eps", sep = ''))
}


arg <- commandArgs(trailingOnly = TRUE)
argLen <- length(arg)
if (argLen == 3) {
    arg_b <- as.integer(arg[1]) # total b bits
    arg_s <- as.integer(arg[2]) # randomly sampling s bits
    dataname <- arg[3] # output filename

    print(paste("Draw ANN figure for multi-probe: b = ", arg_b, ", s = ", arg_s, sep = ''))

    library(combinat)

    multi_probe_levels = c(0:4)
    distances = c(0:4)

    data_frames = rbind()

    for (distance in distances) { 
        accum_nn_percentage = 0
        print(paste("d = ", distance, sep = ''))

        for (level in multi_probe_levels) {
            print(paste("m = ", level, sep = ''))

            i = c((distance - level):0)

            rCi = sum(nCm(arg_b - arg_s, i[i >= 0]))

            sCm = nCm(arg_s, level)
            bCj = sum(nCm(arg_b, c(distance:0)))

            accum_nn_percentage = accum_nn_percentage + sCm * rCi / bCj

            if (accum_nn_percentage > 1.0) {
                accum_nn_percentage = 1.0
            } 
 
            data_frames = rbind(data.frame(probe = level, nn = accum_nn_percentage * 100, distance = paste("d = ", distance, sep = '')), data_frames)
        }
    }

    print(data_frames)

    draw_figure_ggplot(dataname, data_frames)
}

