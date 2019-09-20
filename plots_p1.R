library(ggplot2)
library(tidyr)
library(stringr)
library(dplyr)

## Figure 3
data <- read.csv("results/prob1_aps_psia.csv", col.names = 0:9, check.names = FALSE)
data_n <- data %>% 
  gather(key = "d", value = "a", convert = TRUE) %>%
  mutate(d = factor(d), a = factor(a)) %>%
  group_by(d, a) %>%
  summarize(Density = n()/nrow(.))

p <- ggplot(data_n, aes(x = d, y = Density, fill = a)) + 
  geom_col(position = "dodge") + theme_bw()

ggsave(p, filename = "img/prob1_aps_psia.pdf", dpi = 300)


data <- read.csv("results/prob1_mc_psia.csv", 
                 col.names = c(0, 1), 
                 check.names = FALSE)

data_n <- data %>%
  mutate(d = factor(0:9)) %>%
  gather(-d, key = "a", value = "Density", convert = TRUE) %>%
  mutate(a = factor(a))
  
p <- ggplot(data_n, aes(x = d, y = Density, fill = a)) + 
  geom_col(position = "dodge") + theme_bw()

ggsave(p, filename = "img/prob1_mc_psia.pdf", dpi = 300)


## Figure 4
dist <- read.csv("results/prob1_aps_psid.csv", col.names = "d")
dens <- count(dist, d)
dens$freq <- dens$n/nrow(dist)

p <- ggplot(dens, aes(x=d, y=freq))+ 
  geom_bar(stat="identity", colour="black", fill = "white") + 
  theme_bw() + 
  xlab("Optimal Decision") + 
  ylab("Density") + 
  scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0))

ggsave(p, filename = "img/prob1_aps_psid.pdf", dpi = 300)


dist <- read.csv("results/prob1_mc_psid.csv")
dist$d <- 0:9

p <- ggplot(dist, aes(x=d, y=psi_d)) + 
  geom_bar(stat="identity", colour="black", fill = "white") +  
  theme_bw() + 
  xlab("Optimal Decision") + 
  ylab("Expected Utility") + 
  scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0))

ggsave(p, filename = "img/prob1_mcmc_psid.pdf", dpi = 300)

## Figure 6

aps <- read.csv('./results/prob1_aps_pa_ara.csv', 
                 col.names = c(0, 1),
                 check.names = FALSE) %>% mutate(d = factor(0:9))

mc <- read.csv('./results/prob1_mc_pa_ara.csv',  
                 col.names = c(0, 1),
                 check.names = FALSE) %>% mutate(d = factor(0:9))

############################## SEPARATE PLOTS######################################################
###################################################################################################
data_n <- aps %>%
  mutate(d = factor(0:9)) %>%
  gather(-d, key = "a", value = "Density", convert = TRUE) %>%
  mutate(a = factor(a))

p <- ggplot(data_n, aes(x = d, y = Density, fill = a)) + 
  geom_col(position = "dodge") + theme_bw()

ggsave(p, filename = "img/prob1_pa_ara_aps.pdf", dpi = 300, width = 14, height = 7)

data_n <- mc %>%
  mutate(d = factor(0:9)) %>%
  gather(-d, key = "a", value = "Density", convert = TRUE) %>%
  mutate(a = factor(a))

p <- ggplot(data_n, aes(x = d, y = Density, fill = a)) + 
  geom_col(position = "dodge") + theme_bw()

ggsave(p, filename = "img/prob1_pa_ara_mc.pdf", dpi = 300, width = 14, height = 7)

############################## JOIN PLOTS######################################################
###################################################################################################

data <- bind_rows(APS = gather(aps, -d, key = "a", value = "pa"),
                  MC  = gather( mc, -d, key = "a", value = "pa"),
                  .id = "Algorithm")

p <- ggplot(data, aes(x = d, fill = a, y = pa)) + 
  geom_col(position = "dodge") + 
  facet_wrap(vars(Algorithm)) + theme_bw() 

ggsave(p, filename = "img/prob1_pa_ara.pdf", dpi = 300, width = 14, height = 7)


## Figure 7
dist <- read.csv("results/prob1_aps_psid_ara.csv", col.names = "d")
dens <- count(dist, d)
dens$freq <- dens$n/nrow(dist)

p <- ggplot(dens, aes(x=d, y=freq))+ 
  geom_bar(stat="identity", colour="black", fill = "white") + 
  theme_bw() + 
  xlab("Optimal Decision") + 
  ylab("Density") + 
  scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0))

ggsave(p, filename = "img/prob1_aps_psid_ara.pdf", dpi = 300)


dist <- read.csv("results/prob1_mc_psid_ara.csv")
dist$d <- 0:9

p <- ggplot(dist, aes(x=d, y=psi_d)) + 
  geom_bar(stat="identity", colour="black", fill = "white") +  
  theme_bw() + 
  xlab("Optimal Decision") + 
  ylab("Expected Utility") + 
  scale_x_continuous(limits = c(-1,10), breaks = seq(0, 9, by = 1), expand=c(0,0))

ggsave(p, filename = "img/prob1_mc_psid_ara.pdf", dpi = 300)
