library(ggplot2)
library(reshape2)
library(plyr)

## Problem 1 - APS

dist = read.csv("results/prob1_aps_psid.csv")
dens = count(dist)
dens$freq = dens$freq/nrow(dist)

p = ggplot(dens, aes(x=samps, y=freq))
p = p + geom_bar(stat="identity", colour="black", fill = "white")
p = p +  theme_bw() + xlab("Optimal Decision") + ylab("Density")
p = p + scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0))
p
ggsave(p, filename = "img/prob1_aps_psid.eps", device = "eps", dpi = 300)

####

dist = read.csv("results/prob1_mc_psid.csv")
dist$decision = 0:9

p = ggplot(dist, aes(x=decision, y=psi_d))
p = p + geom_bar(stat="identity", colour="black", fill = "white")
p = p +  theme_bw() + xlab("Optimal Decision") + ylab("Expected Utility")
p = p + scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0))
p

ggsave(p, filename = "img/prob1_mcmc_psid.eps", device = "eps", dpi = 300)

###

psia = read.csv("results/prob1_aps_psia.csv")
## Sensitivity Analysis
dist = read.csv("results/sa_results.csv")


p = ggplot(dist)
p = p + geom_histogram(aes(x = pert_dec, y = ..density..), colour="black", fill = "white", bins = 12)
p = p +  theme_bw() + xlab("Optimal Decision") + ylab("Density")
p = p + scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0))
p

ggsave(p, filename = "img/SA.eps", device = "eps", dpi = 300)
