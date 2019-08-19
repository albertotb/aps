library(ggplot2)
library(reshape2)
## APS solution real problem
dist = read.csv("results/dist_APS.csv")
p = ggplot(dist, aes(x = X0)) + geom_histogram(aes(y=(..count..)/sum(..count..)), bins = 40, colour="black", fill="white")
p = p + geom_vline(aes(xintercept=125),
                   color="blue", linetype="dashed", size=1)
p = p +  theme_bw() + xlab("Defender's Decision") + ylab("Frequency")
p

ggsave(p, filename = "figs/aps_prob3.eps", device = "eps", dpi = 300)

## Expected utility time comparison problem

#  pd.DataFrame({"d": d_values, "EU": psi_d})

psi_d = read.csv("results/EU_timecompprob.csv")
p = ggplot(psi_d, aes(x = d, y = EU)) + geom_bar(stat="identity", colour="black", fill = "white") 
p = p +  theme_bw() + xlab("Defender's Decision") + ylab("Defender's Expected Utility")
p

ggsave(p, filename = "figs/EU_probtime.eps", device = "eps", dpi = 300)


## Cost interpolation

#  pd.DataFrame({"d": d_values, "EU": psi_d})

costs = read.csv("results/costs.csv")
costs = costs[costs$x <= 200,]
p = ggplot(costs, aes(x = x, y = y)) + geom_line()
p = p +  theme_bw() + xlab("Amount of protection in gbps") + ylab("Cost in Euros")
p

ggsave(p, filename = "figs/costs.eps", device = "eps", dpi = 300)

## p_d real problem

pd = read.csv("results/p_d.csv")
colnames(pd) = c("d", 0:30)
pd_melted = melt(pd, id.vars = "d")
ds = c(0, 50, 100, 195)
pd_melted = pd_melted[pd_melted$d %in% ds,]
p = ggplot(data = pd_melted, aes(x = variable, y = value)) + geom_bar(stat = "identity", colour="black", fill = "white")
p = p + facet_wrap(~ d, nrow = 2)
p = p + xlab("Attacker's Decision") + ylab("p(a|d)") + theme_bw()
ggsave(p, filename = "figs/pa_given_d.eps", device = "eps", dpi = 300)
