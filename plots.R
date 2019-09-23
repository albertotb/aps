library(ggplot2)
library(reshape2)

## Sensitivity Analysis
dist = read.csv("results/sa_results.csv")

p <- ggplot(dist, aes(x = pert_dec)) + 
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge", color = "black", fill = "white") +
  xlab("Optimal Decision") + 
  ylab("Density")

ggsave(p, filename = "img/hist_sa2.pdf", dpi = 300)


## APS solution real problem
dist <- read.csv("results/dist_APS.csv")
p <- ggplot(dist, aes(x = X0,
                      fill = factor(ifelse(X0 == 125, "Highlighted", "Normal")))) + 
  geom_histogram(aes(y=(..count..)/sum(..count..)), bins = 40, colour="black") +
  scale_fill_manual(name = "area", values=c("red", "white"), guide = FALSE) +
  scale_x_continuous(breaks = c(0, 50, 100, 125, 150, 200)) +
  xlab("Defender's Decision") + 
  ylab("Frequency")
p

ggsave(p, filename = "img/aps_prob3.pdf", device = "pdf", dpi = 300)

## Expected utility time comparison problem
psi_d <- read.csv("results/EU_timecompprob.csv")

p <- ggplot(psi_d, aes(x = d, y = EU)) + 
  geom_bar(stat="identity", colour="black", fill = "white") + 
  xlab("Defender's Decision") + 
  ylab("Defender's Expected Utility")

ggsave(p, filename = "img/EU_probtime.pdf", device = "pdf", dpi = 300)

## Cost interpolation
costs <- read.csv("results/costs.csv")
costs <- costs[costs$x <= 200,]

p <- ggplot(costs, aes(x = x, y = y)) + 
  geom_line() + 
  xlab("Amount of protection in gbps") + 
  ylab("Cost in Euros")

ggsave(p, filename = "img/costs.pdf", device = "pdf", dpi = 300)

## p_d real problem
pd <- read.csv("results/p_d.csv")
colnames(pd) <- c("d", 0:30)
pd_melted <- melt(pd, id.vars = "d")
ds <- c(0, 50, 100, 195)
pd_melted <- pd_melted[pd_melted$d %in% ds,]

p <- ggplot(data = pd_melted, aes(x = variable, y = value)) + 
  geom_bar(stat = "identity", colour="black", fill = "white") + 
  scale_x_discrete(breaks = seq(0, 30, 2)) + 
  facet_wrap(~ d, nrow = 2) + 
  xlab("Attacker's Decision") + 
  ylab("p(a|d)")

ggsave(p, filename = "img/pa_given_d.pdf", device = "pdf", dpi = 300)
