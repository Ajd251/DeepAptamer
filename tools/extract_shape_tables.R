suppressPackageStartupMessages(library(DNAshapeR))
set.seed(7)
B <- c("A","C","G","T")
pent <- apply(expand.grid(B,B,B,B,B), 1, paste0, collapse="")
rnd <- function(n) paste(sample(B, n, replace=TRUE), collapse="")
seqs <- c(replicate(1500, rnd(100)),
          sapply(pent, function(p) paste0(p, rnd(95))),
          sapply(pent, function(p) paste0(rnd(95), p)))
stopifnot(all(nchar(seqs) == 100))
writeLines(paste0(">s", seq_along(seqs), "\n", seqs), "p3.fa")
p <- getShape("p3.fa", shapeType=c("MGW","ProT","Roll","HelT"))
for (nm in names(p)) write.table(p[[nm]], paste0("p3_", nm, ".tsv"), sep="\t",
                                 row.names=FALSE, col.names=FALSE, na="NA")
writeLines(seqs, "p3_seqs.txt")
cat("sequences:", length(seqs), "all length 100\n")
