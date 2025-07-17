if (!require("CORElearn")) {
  install.packages("CORElearn", repos = "http://cran.us.r-project.org")
}
library(CORElearn)

cat("CORElearn package loaded.\n")

X <- as.matrix(read.csv("benchmark_X.csv", header = FALSE))
y <- read.csv("benchmark_y.csv", header = FALSE)$V1

benchmark_data <- as.data.frame(X)
benchmark_data$target <- as.factor(y)

cat("Data loaded and prepared.\n\n")


k_neighbors <- 10

cat(paste("Starting CORElearn benchmark (ReliefF with k =", k_neighbors, ")...\n"))

timing <- system.time({
  scores <- attrEval(target ~ ., data = benchmark_data,
                     estimator = "ReliefFequalK",
                     ReliefIterations = k_neighbors)
})

cat("Benchmark finished.\n\n")

cat("--- R CORElearn Execution Time ---\n")
print(timing)

cat("\n--- Top 5 Feature Scores ---\n")
print(head(sort(scores, decreasing = TRUE), 5))