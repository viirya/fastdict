
draw_figure <- function(dataname) {

    data <- read.csv(paste(dataname, ".csv", sep = ''), header = FALSE, sep = ' ')
    print(data)
    png(filename = paste(dataname, ".png", sep = ''))

    plot(data[, 1], data[, 2], type = "l", xlim = c(0, 4), ylim = c(0, 100), xlab = "multi-probe number", ylab = "NN percentage", cex = 1.5, col = "red")
    lines(data[, 1], data[, 3], col = rgb(0, 0, 1))
    lines(data[, 1], data[, 4], col = rgb(0, 1, 1))
    lines(data[, 1], data[, 5], col = rgb(0, 1, 0))
}

draw_figure_ggplot <- function(dataname, ann_data_frames) {

    library(ggplot2)
 
    data <- read.csv(paste(dataname, ".csv", sep = ''), header = FALSE, sep = ' ')
    print(data)

    data_frames = rbind()
    for (i in 2:ncol(r32)) {
        data_frames = rbind(data.frame(probe = r32[, 1], nn = r32[, i], distance = paste("d = ", i - 2, sep = '')), data_frames)
    }

    print(data_frames)

    p <- ggplot(data_frames, aes(x = probe, y = nn, group = distance))
    p + geom_line(aes(colour = distance)) + scale_colour_discrete(h = c(0, 360) + 15, c = 100, h.start = 0, direction = 1) + xlab("Number of multi-probe") + ylab("NN percentage") + theme_bw() + theme(axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15), legend.text = element_text(size = 15), legend.title = element_text(size = 15), axis.text = element_text(size = 15)) + geom_line(data = ann_data_frames, aes(x = probe, y = nn, group = distance, colour = distance), linetype = "dashed", show_guide = FALSE)

    ggsave(file = paste(dataname, ".eps", sep = ''))
}


arg <- commandArgs(trailingOnly = TRUE)
argLen <- length(arg)
if (argLen == 3) {
    datafile <- arg[1]
    print(paste("Draw multi-probe figure for ", arg, "...", sep = ''))
 
    arg_b <- as.integer(arg[2]) # total b bits
    arg_s <- as.integer(arg[3]) # randomly sampling s bits

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
 
    draw_figure_ggplot(datafile, data_frames)
}

