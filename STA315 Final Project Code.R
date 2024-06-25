
#Package Installation
install.packages("glmnet")
install.packages("caret")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("pracma")


library(pracma)
library(glmnet)
library(dplyr)
library(caret)
library(ggplot2)
set.seed(1005038166)

#Parameters for matrix
n = 21
p = 10
ValueRatio = 1.54 * 10^5
NormXBeta2 = 370.84

#Generate Matrix
X = matrix(rnorm(n * p), nrow = n, ncol = p)

svd_decomposition = svd(X)
U = svd_decomposition$u
V = svd_decomposition$v
SingularValues = svd_decomposition$d
#Make sure the ratio of largest to smallest is 1.54x10^5
SingularValues[1] = SingularValues[p] * ValueRatio

XMod = U %*% diag(SingularValues) %*% t(V)

Scaling = sqrt(NormXBeta2 / norm(XMod %*% rep(1, p))^2)
XFinal = XMod * Scaling

#Vector containing the 4 methods that will be assessed
cv_methods = c("gcv", "re", "mle", "apress")
monte_carlo_runs = 100

#Vector to hold results of the inefficiencies
results = data.frame(Method = character(), Run = integer(), Lambda = double(), Inefficiency = double(), stringsAsFactors = FALSE)

#Finds noise value for response variable
snr = 4200
noise_sd = sqrt((norm(XFinal %*% rep(1, p))^2) / snr)

for (run in 1:monte_carlo_runs) {
  #Generate a new response variable y for each run
  beta = rnorm(p)
  noise = rnorm(n, mean = 0, sd = noise_sd)
  Y = XFinal %*% beta + noise
  
  for (method in cv_methods) {
    ridge_cv = cv.glmnet(XFinal, Y, alpha = 0, nfolds = 10, type.measure = "mse", lambda.min.ratio = 1e-10, lambda.type = method)
    best_lambda = ridge_cv$lambda.min
    best_model = glmnet(XFinal, Y, alpha = 0, lambda = best_lambda)
    inefficiency = mean((Y - XFinal %*% coef(best_model)[-1])^2)
    
    results = rbind(results, data.frame(Method = method, Run = run, Lambda = best_lambda, Inefficiency = inefficiency))
  }
}

#Calculate the inefficiencies for each method
AvgIneff = results %>% group_by(Method) %>% summarise(AvgIneff = mean(Inefficiency))

#Orders the results from lowest to highest
AvgIneff = AvgIneff[order(AvgIneff$AvgIneff),]

#Prints table
print.data.frame(AvgIneff, digits = 4)
