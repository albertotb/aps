library(ggplot2)
library(reshape2)

## Sensitivity Analysis
dist = read.csv("results/sa_results.csv")


p <- ggplot(dist, aes(x = pert_dec)) + 
  geom_bar(aes(y=..density..), position = "dodge", width = 1) +
  scale_y_continuous(limits=c(0,1)) + 
  xlab("Optimal Decision") + 
  ylab("Density")

ggsave(p, filename = "img/sa.pdf", dpi = 300)


## APS solution real problem
dist = read.csv("results/dist_APS.csv")
p = ggplot(dist, aes(x = X0)) + geom_histogram(aes(y=(..count..)/sum(..count..)), bins = 40, colour="black", fill="white")
p = p + geom_vline(aes(xintercept=125),
                   color="blue", linetype="dashed", size=1)
p = p + xlab("Defender's Decision") + ylab("Frequency")
p

ggsave(p, filename = "img/aps_prob3.pdf", device = "pdf", dpi = 300)

## Expected utility time comparison problem

#  pd.DataFrame({"d": d_values, "EU": psi_d})

psi_d = read.csv("results/EU_timecompprob.csv")
p = ggplot(psi_d, aes(x = d, y = EU)) + geom_bar(stat="identity", colour="black", fill = "white") 
p = p + xlab("Defender's Decision") + ylab("Defender's Expected Utility")
p

ggsave(p, filename = "img/EU_probtime.pdf", device = "pdf", dpi = 300)


## Cost interpolation

#  pd.DataFrame({"d": d_values, "EU": psi_d})

costs = read.csv("costs.csv")
costs = costs[costs$x <= 200,]
p = ggplot(costs, aes(x = x, y = y)) + geom_line()
p = p + xlab("Amount of protection in gbps") + ylab("Cost in Euros")
p

ggsave(p, filename = "img/costs.pdf", device = "pdf", dpi = 300)

## p_d real problem

pd = read.csv("results/p_d.csv")
colnames(pd) = c("d", 0:30)
pd_melted = melt(pd, id.vars = "d")
ds = c(0, 50, 100, 195)
pd_melted = pd_melted[pd_melted$d %in% ds,]
p = ggplot(data = pd_melted, aes(x = variable, y = value)) + geom_bar(stat = "identity", colour="black", fill = "white")
p = p + facet_wrap(~ d, nrow = 2)
p = p + xlab("Attacker's Decision") + ylab("p(a|d)")
ggsave(p, filename = "img/pa_given_d.pdf", device = "pdf", dpi = 300)
