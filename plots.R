#!/usr/bin/env Rscript

library(ggplot2)
library(tidyr)
library(stringr)
library(dplyr)
library(latex2exp)

## Figure 3
data <- read.csv("results/prob1_aps_psia.csv", col.names = 0:9, check.names = FALSE)
data_n <- data %>% 
  gather(key = "d", value = "a", convert = TRUE) %>%
  mutate(d = factor(d), a = factor(a)) %>%
  group_by(d, a) %>%
  summarize(Frequency = n()/nrow(data)) ## Watch out! This is nrow(data)

p <- ggplot(data_n, aes(x = d, y = Frequency, fill = a)) + 
  geom_col(position = "dodge", color="black")
p <- p + scale_fill_manual(values=c("black", "white"))
p
ggsave(p, filename = "img/prob1_aps_psia.pdf", dpi = 300, width = 8.33, height = 5.79)

data <- read.csv("results/prob1_mc_psia.csv", 
                 col.names = c(0, 1), 
                 check.names = FALSE)

data_n <- data %>%
  mutate(d = factor(0:9)) %>%
  gather(-d, key = "a", value = "Expected_Utility", convert = TRUE) %>%
  mutate(a = factor(a))

p <- ggplot(data_n, aes(x = d, y = Expected_Utility, fill = a)) + 
  geom_col(position = "dodge", color="black")
p <- p + scale_fill_manual(values=c("black", "white"))
p = p + ylab("Expected Utility")

ggsave(p, filename = "img/prob1_mc_psia.pdf", dpi = 300, width = 8.33, height = 5.79)


## Figure 4
dist <- read.csv("results/prob1_aps_psid.csv", col.names = "d")
dens <- count(dist, d)
dens$freq <- dens$n/nrow(dist)

p <- ggplot(dens, aes(x=d, y=freq))+ 
  geom_bar(stat="identity", colour="black", fill = "white") + 
  xlab("Optimal Decision") + 
  ylab("Frequency") + 
  scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0))

ggsave(p, filename = "img/prob1_aps_psid.pdf", dpi = 300, width = 8.33, height = 5.79)

dist <- read.csv("results/prob1_mc_psid.csv")
dist$d <- 0:9

p <- ggplot(dist, aes(x=d, y=psi_d)) + 
  geom_bar(stat="identity", colour="black", fill = "white") +  
  xlab("Optimal Decision") + 
  ylab("Expected Utility") + 
  scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0))

ggsave(p, filename = "img/prob1_mcmc_psid.pdf", dpi = 300, width = 8.33, height = 5.79)


## Figure 6
aps <- read.csv('./results/prob1_aps_pa_ara.csv', 
                col.names = c(0, 1),
                check.names = FALSE) %>% mutate(d = factor(0:9))

mc <- read.csv('./results/prob1_mc_pa_ara.csv',  
               col.names = c(0, 1),
               check.names = FALSE) %>% mutate(d = factor(0:9))

############################## SEPARATE PLOTS########
data_n <- aps %>%
  mutate(d = factor(0:9)) %>%
  gather(-d, key = "a", value = "Frequency", convert = TRUE) %>%
  mutate(a = factor(a))

exp = TeX("p_D(a|d)")
p <- ggplot(data_n, aes(x = d, y = Frequency, fill = a)) + 
  geom_col(position = "dodge", color="black") + ylab(exp) 

p <- p + scale_fill_manual(values=c("black", "white"))
p
ggsave(p, filename = "img/prob1_pa_ara_aps.pdf", dpi = 300, width = 8.33, height = 5.79)

data_n <- mc %>%
  mutate(d = factor(0:9)) %>%
  gather(-d, key = "a", value = "Expected_Utility", convert = TRUE) %>%
  mutate(a = factor(a))

p <- ggplot(data_n, aes(x = d, y = Expected_Utility, fill = a)) + 
  geom_col(position = "dodge", color="black") + ylab(exp) 
p <- p + scale_fill_manual(values=c("black", "white"))
p
ggsave(p, filename = "img/prob1_pa_ara_mc.pdf", dpi = 300, width = 8.33, height = 5.79)


## Figure 7
dist <- read.csv("results/prob1_aps_psid_ara.csv", col.names = "d")
dens <- count(dist, d)
dens$freq <- dens$n/nrow(dist)

p <- ggplot(dens, aes(x=d, y=freq))+ 
  geom_bar(stat="identity", colour="black", fill = "white") +
  xlab("Optimal Decision") + 
  ylab("Frequency") + 
  scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0))

ggsave(p, filename = "img/prob1_aps_psid_ara.pdf", dpi = 300, width = 8.33, height = 5.79)

dist <- read.csv("results/prob1_mc_psid_ara.csv")
dist$d <- 0:9

p <- ggplot(dist, aes(x=d, y=psi_d)) + 
  geom_bar(stat="identity", colour="black", fill = "white") +  
  xlab("Optimal Decision") + 
  ylab("Expected Utility") + 
  scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0))

ggsave(p, filename = "img/prob1_mc_psid_ara.pdf", dpi = 300, width = 8.33, height = 5.79)


## Sensitivity Analysis
dist = read.csv("results/sa_results.csv")

p <- ggplot(dist, aes(x = pert_dec)) + 
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge", color = "black", fill = "white") +
  xlab("Optimal Decision") + 
  scale_x_continuous(breaks = 0:9) +
  ylab("Density")

ggsave(p, filename = "img/hist_sa2.pdf", dpi = 300, width = 8.33, height = 5.79)


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

ggsave(p, filename = "img/aps_prob3.pdf", device = "pdf", dpi = 300, width = 8.33, height = 5.79)


## Expected utility time comparison problem
psi_d <- read.csv("results/EU_timecompprob.csv")

p <- ggplot(psi_d, aes(x = d, y = EU)) + 
  geom_bar(stat="identity", colour="black", fill = "white") + 
  xlab("Defender's Decision") + 
  ylab("Defender's Expected Utility")

ggsave(p, filename = "img/EU_probtime.pdf", device = "pdf", dpi = 300, width = 8.33, height = 5.79)


## Cost interpolation
costs <- read.csv("results/costs.csv")
costs <- costs[costs$x <= 200,]

p <- ggplot(costs, aes(x = x, y = y)) + 
  geom_line() + 
  xlab("Amount of protection in gbps") + 
  ylab("Cost in Euros")

ggsave(p, filename = "img/costs.pdf", device = "pdf", dpi = 300, width = 8.33, height = 5.79)


## p_d real problem
pd <- read.csv("results/p_d.csv", col.names = c("d", 0:30), check.names = FALSE)

pd_melted <- pd %>%
  gather(-d, key = "variable", value = "value", convert = TRUE) %>%
  filter(d %in% c(0, 50, 100, 195))

p <- ggplot(data = pd_melted, aes(x = factor(variable), y = value)) + 
  geom_bar(stat = "identity", colour="black", fill = "white") + 
  scale_x_discrete(breaks = seq(0, 30, 2)) + 
  facet_wrap(~ d, nrow = 2) + 
  xlab("Attacker's Decision") + 
  ylab("p(a|d)")

ggsave(p, filename = "img/pa_given_d.pdf", device = "pdf", dpi = 300, width = 8.33, height = 5.79)
