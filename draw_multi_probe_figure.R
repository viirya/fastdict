
draw_figure <- function(dataname) {

    data <- read.csv(paste(dataname, ".csv", sep = ''), header = FALSE, sep = ' ')
    print(data)
    png(filename = paste(dataname, ".png", sep = ''))

    plot(data[, 1], data[, 2], type = "l", xlim = c(0, 4), ylim = c(0, 100), xlab = "multi-probe number", ylab = "NN percentage", cex = 1.5, col = "red")
    lines(data[, 1], data[, 3], col = rgb(0, 0, 1))
    lines(data[, 1], data[, 4], col = rgb(0, 1, 1))
    lines(data[, 1], data[, 5], col = rgb(0, 1, 0))
}

draw_figure_ggplot <- function(dataname) {

    library(ggplot2)
 
    data <- read.csv(paste(dataname, ".csv", sep = ''), header = FALSE, sep = ' ')
    print(data)

    #png(filename = paste(dataname, ".png", sep = ''))
    
    data_frames = rbind()
    for (i in 2:ncol(r32)) {
        data_frames = rbind(data.frame(probe = r32[, 1], nn = r32[, i], distance = paste("d = ", i - 2, sep = '')), data_frames)
    }

    print(data_frames)

    p <- ggplot(data_frames, aes(x = probe, y = nn, group = distance))
    p + geom_line(aes(colour = distance)) + scale_colour_discrete(h = c(0, 360) + 15, c = 100, h.start = 0, direction = 1) + xlab("Number of multi-probe") + ylab("NN percentage") + theme_bw() + theme(axis.title.x = element_text(size = 15), axis.title.y = element_text(size = 15), legend.text = element_text(size = 15), legend.title = element_text(size = 15), axis.text = element_text(size = 15))

    ggsave(file = paste(dataname, ".eps", sep = ''))
}


arg <- commandArgs(trailingOnly = TRUE)
argLen <- length(arg)
if (argLen == 1) {
    arg <- arg[argLen]
    print(paste("Draw multi-probe figure for ", arg, "...", sep = ''))
    draw_figure_ggplot(arg)
}

