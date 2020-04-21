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

# DATA
path <- "./results/prob1"
d3a <- read_csv(file.path(path, "1586784517_prob1_mcmc_adg_psia.csv"), col_types = c(d = "c"))
d3b <- read.csv(file.path(path, "1586798776_prob1_aps_adg_psia.csv"), col.names = 0:9, check.names = FALSE)

d4a <- read_csv(file.path(path, "1586784517_prob1_mcmc_adg_psid.csv"), col_types = c(d = "c"))
d4b <- read.csv(file.path(path, "1586798776_prob1_aps_adg_psid.csv"), col.names = "d")

d6a <- read_csv(file.path(path, "1586877226_prob1_mcmc_ara_pa.csv"))
d6b <- read_csv(file.path(path, "1586874387_prob1_aps_ara_pa.csv"))

d7a <- read.csv(file.path(path, "1586865337_prob1_mcmc_ara_psid.csv"), col.names = c("d", "mean"))
d7b <- read.csv(file.path(path, "1586865621_prob1_aps_ara_psid.csv"), col.names = "d")

#---------------------------------------------------------------------------------------

# Data figure 3
d3a_long <- pivot_longer(d3a, -d, 
                         names_to = c(".value", "a"), 
                         names_pattern = "(.*)_(.*)")

d3b_long <- d3b %>% 
  gather(key = "d", value = "a", convert = TRUE) %>%
  mutate(d = factor(d), a = factor(a)) %>%
  group_by(d, a) %>%
  summarize(Frequency = n()/nrow(d3b))

# Figure 3
p3a <- ggplot(d3a_long, aes(x = d, y = mean, fill = a)) +
  geom_col(position = "dodge", color = "black") +
  #geom_errorbar(aes(ymin = mean-std, ymax = mean+std), width=.2, position=position_dodge(.9)) +
  scale_fill_manual(values=colors) +
  ylab("Expected Utility") +
  theme(text = element_text(size=20))

p3b <- ggplot(d3b_long, aes(x = d, y = Frequency, fill = a)) +
  geom_col(position = "dodge", color = "black") +
  scale_fill_manual(values = colors) + 
  theme(text = element_text(size=20))

ggsave(p3a, filename = "img/prob1_mc_psia.pdf", dpi = dpi, width = width, height = height)
ggsave(p3b, filename = "img/prob1_aps_psia.pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

## Data figure 4
d4b <- count(d4b, d)
d4b$freq <- d4b$n/nrow(d4b)

# Figure 4
p4a <- ggplot(d4a, aes(x=d)) +
  geom_col(aes(y = mean), color = "black", fill = colors[2]) +
  #geom_errorbar(aes(ymax = mean+std, ymin = mean-std), width=.2) +
  xlab("Optimal Decision") +
  ylab("Expected Utility") +
  theme(text = element_text(size=20))

p4b <- ggplot(d4b, aes(x=d, y=freq))+
  geom_bar(stat="identity", color = "black", fill = colors[2]) +
  xlab("Optimal Decision") +
  ylab("Frequency") +
  scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0)) + 
  theme(text = element_text(size=20))

ggsave(p4a, filename = "img/prob1_mc_psid.pdf", dpi = dpi, width = width, height = height)
ggsave(p4b, filename = "img/prob1_aps_psid.pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

# Data figure 6
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
  scale_fill_manual(values = colors) + 
  theme(text = element_text(size=20))

p6b <- ggplot(d6b_long, aes(x = d, y = Frequency, fill = a)) +
  geom_col(position = "dodge", color = "black") +
  ylab(exp) +
  scale_fill_manual(values = colors) + 
  theme(text = element_text(size=20))

ggsave(p6a, filename = "img/prob1_pa_ara_mc.pdf", dpi = dpi, width = width, height = height)
ggsave(p6b, filename = "img/prob1_pa_ara_aps.pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

# Data figure 7
d7b_proc <- count(d7b, d)
d7b_proc$freq <- d7b_proc$n/nrow(d7b)

# Figure 7
p7a <- ggplot(d7a, aes(x = d)) +
  geom_col(aes(y = mean), colour = "black", fill = colors[2]) +
  #geom_errorbar(aes(ymax = mean+std, ymin = mean-std), width = 0.2) +
  xlab("Optimal Decision") +
  ylab("Expected Utility") + 
  theme(text = element_text(size=20))

p7b <- ggplot(d7b_proc, aes(x = d, y = freq))+ 
  geom_bar(stat="identity", colour = "black", fill = colors[2]) +
  xlab("Optimal Decision") +
  ylab("Frequency") +
  scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0)) + 
  theme(text = element_text(size=20))

ggsave(p7a, filename = "img/prob1_mc_psid_ara.pdf", dpi = dpi, width = width, height = height)
ggsave(p7b, filename = "img/prob1_aps_psid_ara.pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

# Sensitivity Analysis
dist <- read.csv("results/sa_results.csv")

p <- ggplot(dist, aes(x = pert_dec)) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position = "dodge", color = "black", fill = colors[2]) +
  xlab("Optimal Decision") +
  scale_x_continuous(breaks = 0:9) +
  ylab("Density") + 
  theme(text = element_text(size=20))

ggsave(p, filename = "img/hist_sa2.pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------
# volver al esquema de colores anterior
#colors <- c("red", "white")
#------------------------------------------------------------------------------

# APS solution real problem
dist <- read.csv("results/dist_APS.csv")
p <- ggplot(dist, aes(x = X0,
                      fill = factor(ifelse(X0 == 125, "Highlighted", "Normal")))) + 
  geom_histogram(aes(y=(..count..)/sum(..count..)), bins = 40, colour = "black") +
  scale_fill_manual(name = "area", values = colors, guide = FALSE) +
  scale_x_continuous(breaks = c(0, 50, 100, 125, 150, 200)) +
  xlab("Defender's Decision") +
  ylab("Frequency") + 
  theme(text = element_text(size=20))

ggsave(p, filename = "img/aps_prob3.pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

# Expected utility time comparison problem
psi_d <- read.csv("results/EU_timecompprob.csv")

p <- ggplot(psi_d, aes(x = d, y = EU)) +
  geom_bar(stat="identity", colour = "black", fill = colors[2]) +
  xlab("Defender's Decision") +
  ylab("Defender's Expected Utility") + 
  theme(text = element_text(size=20))

ggsave(p, filename = "img/EU_probtime.pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

# Cost interpolation
costs <- read.csv("results/costs.csv")
costs <- costs[costs$x <= 200,]

p <- ggplot(costs, aes(x = x, y = y)) +
  geom_line() +
  xlab("Amount of protection in gbps") +
  ylab("Cost in Euros") + 
  theme(text = element_text(size=20))

ggsave(p, filename = "img/costs.pdf", device = "pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

# p_d real problem
pd <- read.csv("results/p_ad.csv", col.names = c("d", 0:30), check.names = FALSE)

pd_melted <- pd %>%
  gather(-d, key = "variable", value = "value", convert = TRUE) %>%
  filter(d %in% c(0, 50, 100, 195))

p <- ggplot(data = pd_melted, aes(x = factor(variable), y = value)) +
  geom_bar(stat = "identity", colour = "black", fill = colors[2]) +
  scale_x_discrete(breaks = seq(0, 30, 2)) +
  facet_wrap(~ d, nrow = 2) +
  xlab("Attacker's Decision") +
  ylab("p(a|d)") + 
  theme(text = element_text(size=20))

ggsave(p, filename = "img/pa_given_d.pdf", dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# APS solution real problem with Temperature
J_grid = seq(1000, 9500, by = 500)
for(J in J_grid){
  path <- paste0("results/dist_APS_J", as.character(J), '.csv')
  path_out <- paste0("img/aps_prob3_J", as.character(J), '.pdf')
  title <- paste('H =', as.character(J))
  dist <- read.csv(path, header = F)
  mode = getmode(dist$V1)
  p <- ggplot(dist, aes(x = V1,
                        fill = factor(ifelse(V1 == mode, "Highlighted", "Normal")))) + 
    geom_histogram(aes(y=(..count..)/sum(..count..)), bins = 41, colour="black") +
    scale_fill_manual(name = "area", values = colors, guide = FALSE) +
    xlim(0,200) +
    xlab("Defender's Decision") +
    ylab("Frequency") + 
    theme(plot.title = element_text(hjust = 0.5)) +
    ggtitle(title) + 
    theme(text = element_text(size=20))
  
  ggsave(p, filename = path_out, dpi = dpi, width = width, height = height)
}

path <- paste0("results/dist_APS_no_Jtrick.csv")
path_out <- paste0('img/aps_prob3_no_Jtrick.pdf')
title <- paste('APS')
dist <- read.csv(path, header = F)
burnin <- round(0.20*length(dist$V1))
dist <- slice(dist, burnin:n())
mode <- 0

p <- ggplot(dist, aes(x = V1,
                      fill = factor(ifelse(V1 == mode, "Highlighted", "Normal")))) + 
  geom_histogram(aes(y=(..count..)/sum(..count..)), bins = 41, colour="black") +
  scale_fill_manual(name = "area", values = colors, guide = FALSE) +
  xlim(0,200) +
  xlab("Defender's Decision") +
  ylab("Frequency")  + 
  theme(text = element_text(size=20))
  # theme(plot.title = element_text(hjust = 0.5)) +
  # ggtitle(title)


ggsave(p, filename = path_out, dpi = dpi, width = width, height = height)

#------------------------------------------------------------------------------

# p_d real problem
pd <- read.csv("results/p_ad.csv", col.names = c("d", 0:30), check.names = FALSE)

pd_melted <- pd %>%
  gather(-d, key = "variable", value = "value", convert = TRUE) %>%
  filter(d %in% c(0, 5, 10, 15))

p <- ggplot(data = pd_melted, aes(x = factor(variable), y = value)) +
  geom_bar(stat = "identity", colour = "black", fill = colors[2]) +
  scale_x_discrete(breaks = seq(0, 30, 2)) +
  facet_wrap(~ d, nrow = 2) +
  xlab("Attacker's Decision") +
  ylab("p(a|d)") + 
  theme(text = element_text(size=20))

ggsave(p, filename = "img/pa_given_d_new.pdf", dpi = dpi, width = width, height = height)


#------------------------------------------------------------------------------
# Electronic Companion for Problem 1
#------------------------------------------------------------------------------
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# APS solution real problem with Temperature
#J_grid = seq(10, 490, by = 10)
J_grid <- c(450)
for(J in J_grid){
  path <- paste0("results/prob1/prob1_adg_peaked", as.character(J), '.csv')
  path_out <- paste0("img/prob1_adg_peaked", as.character(J), '.pdf')
  title <- paste('H =', as.character(J))
  dist <- read.csv(path, header = F)
  mode = getmode(dist$V1)
  count_dist <- count(dist, V1)
  count_dist$freq <- count_dist$n/nrow(count_dist)
  p <- ggplot(count_dist, aes(x=V1, y=freq, 
                              fill = factor(ifelse(V1 == mode, "Highlighted", "Normal"))) )+
    geom_bar(stat="identity", colour = "black") +
    scale_fill_manual(name = "area", values = colors, guide = FALSE) +
    xlab("Defender's Decision") +
    ylab("Frequency") +
    scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0)) +
    theme(plot.title = element_text(hjust = 0.5), text = element_text(size=20)) +
    ggtitle(title) + 
    
  ggsave(p, filename = path_out, dpi = dpi, width = width, height = height)
}

#------------------------------------------------------------------------------


# APS solution real problem with Temperature
#J_grid = seq(10, 490, by = 10)
J_grid <- c(450)
for(J in J_grid){
  path <- paste0("results/prob1/prob1_ara_peaked", as.character(J), '.csv')
  path_out <- paste0("img/prob1_ara_peaked", as.character(J), '.pdf')
  title <- paste('H =', as.character(J))
  dist <- read.csv(path, header = F)
  mode = getmode(dist$V1)
  count_dist <- count(dist, V1)
  count_dist$freq <- count_dist$n/nrow(count_dist)
  p <- ggplot(count_dist, aes(x=V1, y=freq, 
                              fill = factor(ifelse(V1 == mode, "Highlighted", "Normal"))) )+
    geom_bar(stat="identity", colour = "black") +
    scale_fill_manual(name = "area", values = colors, guide = FALSE) +
    xlab("Defender's Decision") +
    ylab("Frequency") +
    scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0)) +
    theme(plot.title = element_text(hjust = 0.5), text = element_text(size=20)) +
    ggtitle(title)
  
  ggsave(p, filename = path_out, dpi = dpi, width = width, height = height)
}

#------------------------------------------------------------------------------
