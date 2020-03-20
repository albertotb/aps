#!/usr/bin/env Rscript

library(ggplot2)
library(tidyr)
library(stringr)
library(dplyr)
library(latex2exp)
library(readr)

dpi <- 300
width <- 8.33
height <- 5.79
colors <- c('#ffffb3', '#bebada')

# Data figure 3
d3a <- read_csv("./results/1584635625_prob1_mcmc_adg_psia.csv", col_types = c(d = "c"))
d3a_long <- pivot_longer(d3a, -d, 
                         names_to = c(".value", "a"), 
                         names_pattern = "(.*)_(.*)")

d3b <- read.csv("results/prob1_aps_psia.csv", col.names = 0:9, check.names = FALSE)
d3b_long <- d3b %>% 
  gather(key = "d", value = "a", convert = TRUE) %>%
  mutate(d = factor(d), a = factor(a)) %>%
  group_by(d, a) %>%
  summarize(Frequency = n()/nrow(d3b))

# Figure 3
p3a <- ggplot(d3a_long, aes(x = d, y = mean, fill = a)) +
  geom_col(position = "dodge", color="black") +
  geom_errorbar(aes(ymin = mean-std, ymax = mean+std), width=.2, position=position_dodge(.9)) +
  scale_fill_manual(values=colors) +
  ylab("Expected Utility")

p3b <- ggplot(d3b_long, aes(x = d, y = Frequency, fill = a)) +
  geom_col(position = "dodge", color="black") +
  scale_fill_manual(values = colors)

ggsave(p3a, filename = "img/prob1_mc_psia.pdf", dpi = dpi, width = width, height = height)
ggsave(p3b, filename = "img/prob1_aps_psia.pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

## Data figure 4
d4a <- read_csv("./results/1584635625_prob1_mcmc_adg_psid.csv", col_types = c(d = "c"))

dist <- read.csv("results/prob1_aps_psid.csv", col.names = "d")
d4b <- count(dist, d)
d4b$freq <- d4b$n/nrow(dist)

# Figure 4
p4a <- ggplot(d4a, aes(x=d)) +
  geom_col(aes(y = mean)) +
  geom_errorbar(aes(ymax = mean+std, ymin = mean-std), width=.2) +
  xlab("Optimal Decision") +
  ylab("Expected Utility")

p4b <- ggplot(d4b, aes(x=d, y=freq))+
  geom_bar(stat="identity") +
  xlab("Optimal Decision") +
  ylab("Frequency") +
  scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0))

ggsave(p4a, filename = "img/prob1_mcmc_psid.pdf", dpi = dpi, width = width, height = height)
ggsave(p4b, filename = "img/prob1_aps_psid.pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

# Data figure 6
d6a <- read.csv('./results/prob1_mc_pa_ara.csv', col.names = c(0, 1), check.names = FALSE)
d6b <- read.csv('./results/prob1_aps_pa_ara.csv', col.names = c(0, 1), check.names = FALSE)

d6a_long <- d6a %>%
  mutate(d = factor(0:9)) %>%
  gather(-d, key = "a", value = "Expected_Utility", convert = TRUE) %>%
  mutate(a = factor(a))

d6b_long <- d6b %>%
  mutate(d = factor(0:9)) %>%
  gather(-d, key = "a", value = "Frequency", convert = TRUE) %>%
  mutate(a = factor(a))

# Figure 6
exp = TeX("p_D(a|d)")
p6a <- ggplot(d6a_long, aes(x = d, y = Expected_Utility, fill = a)) +
  geom_col(position = "dodge", color = "black") +
  ylab(exp) +
  scale_fill_manual(values = colors)

p6b <- ggplot(d6b_long, aes(x = d, y = Frequency, fill = a)) +
  geom_col(position = "dodge", color = "black") +
  ylab(exp) +
  scale_fill_manual(values = colors)

ggsave(p6a, filename = "img/prob1_pa_ara_mc.pdf", dpi = dpi, width = width, height = height)
ggsave(p6b, filename = "img/prob1_pa_ara_aps.pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

# Data figure 7
d7a <- read_csv("results/1584637784_prob1_mcmc_ara_psid.csv", col_types = c(d = "c"))

dist <- read.csv("results/prob1_aps_psid_ara.csv", col.names = "d")
d7b <- count(dist, d)
d7b$freq <- d7b$n/nrow(dist)

# Figure 7
p7a <- ggplot(d7a, aes(x = d)) +
  geom_col(aes(y = mean), colour="black", fill = "white") +
  geom_errorbar(aes(ymax = mean+std, ymin = mean-std), width = 0.2) +
  xlab("Optimal Decision") +
  ylab("Expected Utility")

p7b <- ggplot(d7b, aes(x = d, y = freq))+ 
  geom_bar(stat="identity", colour="black", fill = "white") +
  xlab("Optimal Decision") +
  ylab("Frequency") +
  scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0))

ggsave(p7a, filename = "img/prob1_mc_psid_ara.pdf", dpi = dpi, width = width, height = height)
ggsave(p7b, filename = "img/prob1_aps_psid_ara.pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

# Sensitivity Analysis
dist <- read.csv("results/sa_results.csv")

p <- ggplot(dist, aes(x = pert_dec)) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge", color = "black", fill = "white") +
  xlab("Optimal Decision") +
  scale_x_continuous(breaks = 0:9) +
  ylab("Density")

ggsave(p, filename = "img/hist_sa2.pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

# APS solution real problem
dist <- read.csv("results/dist_APS.csv")
p <- ggplot(dist, aes(x = X0,
                      fill = factor(ifelse(X0 == 125, "Highlighted", "Normal")))) + 
  geom_histogram(aes(y=(..count..)/sum(..count..)), bins = 40, colour="black") +
  scale_fill_manual(name = "area", values=c("red", "white"), guide = FALSE) +
  scale_x_continuous(breaks = c(0, 50, 100, 125, 150, 200)) +
  xlab("Defender's Decision") +
  ylab("Frequency")
p

ggsave(p, filename = "img/aps_prob3.pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

# Expected utility time comparison problem
psi_d <- read.csv("results/EU_timecompprob.csv")

p <- ggplot(psi_d, aes(x = d, y = EU)) +
  geom_bar(stat="identity", colour="black", fill = "white") +
  xlab("Defender's Decision") +
  ylab("Defender's Expected Utility")

ggsave(p, filename = "img/EU_probtime.pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

# Cost interpolation
costs <- read.csv("results/costs.csv")
costs <- costs[costs$x <= 200,]

p <- ggplot(costs, aes(x = x, y = y)) +
  geom_line() +
  xlab("Amount of protection in gbps") +
  ylab("Cost in Euros")

ggsave(p, filename = "img/costs.pdf", device = "pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

# p_d real problem
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

ggsave(p, filename = "img/pa_given_d.pdf", dpi = dpi, width = width, height = height)
