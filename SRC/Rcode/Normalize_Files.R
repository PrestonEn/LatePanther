#
# Preston Engstrom
# COSC 4P76
# 
# Normalization of arbitrary CSV data files 


iris = read.csv("data/iris/iris_post.csv", 
                colClasses=c(rep("numeric",4), "character"),
                header = TRUE)

cancer = read.csv("data/wisc/bigcancer_post.csv", 
               colClasses=c(rep("numeric",30), "character"),
               header = TRUE)

norm_cancer <-cancer
norm_iris <-iris

normalizeCol <- function(col){
    min_val = min(col)
    max_val = max(col)
    
    col = (col - min_val)/(max_val - min_val)
    return(col)
}

# Normalize iris data (0-1)
for(i in 1:4){
  norm_iris[i] <- normalizeCol(norm_iris[i])
}

# Normalize cancer data (0-1)
for(i in 1:30){
  norm_cancer[i] <- normalizeCol(norm_cancer[i])
}

write.csv(cancer, file = "data/wisc/cancer_post_norm_0to1.csv", row.names = FALSE, quote = FALSE)
write.csv(iris, file = "data/iris/iris_post_norm_0to1.csv", row.names = FALSE, quote=FALSE)
