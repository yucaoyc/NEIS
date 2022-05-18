# source: https://rdrr.io/cran/spatstat.data/man/finpines.html

library(spatstat)
# convert to csv data
df <- data.frame(x=finpines$x, y=finpines$y, diameter=finpines$marks$diameter, height=finpines$marks$height)
write.csv(df,"finpines.csv")
