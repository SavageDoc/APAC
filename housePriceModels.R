# Include functions used for the Citizens Data Meetup presentation 
# (There are more functions there than we'll use, but fine...)
source('cdmFunctions.R')

# Establish plotting options ----
library( RColorBrewer )
# There will be 3 plots:
# 1) Logistic NN
# 2) Tanh NN
# 3) Average of (1) & (2)
# 4) (Magical) Minimum of the best of 1 & 2
bp <- brewer.pal( n=4, name = 'Dark2' )

## Data setup ----
# Split the data into training/test
myData <- makeSplitData( Boston, trainPercent = 75, splitSeed = 42 )

# Scale the data
trainData <- scaleDF( myData$trainData )
testData <- scaleDF( myData$testData, myData$trainData )

# Setup some parameters:
# Target variable
yName <-'medv'
xNames <- setdiff( names( Boston ), yName )


# Extract scaling parameters
c1 <- min( myData$trainData[,yName] )
c2 <- max( myData$trainData[,yName] ) - min( myData$trainData[,yName] )

# Define the formula for the NN training
myFormula <- buildFormula( yName = yName, xNames = xNames )

## First Nerual Network and assorted plots ----
# Train a neural network - note that the default activation function is 'logistic'
nn1 <- trainNN( trainData=trainData, nnFormula=myFormula, hiddenVec=c(5,3), nnSeed=123 )
scaledPred1 <- compute( nn1, scaleDF( myData$trainData[,xNames]) )$net.result
y1 <- unscalePredictions( scaledPred1, unCentre=c1, unScale=c2 )

# Set up a data frame to hold the true values, predictions and residuals
myDF <- data.frame( Values=myData$trainData[,yName], Pred1=y1, Resid1=y1-myData$trainData[,yName] )

## Scatter plot
scatter1 <- ggplot( data=myDF, aes( x=Values ) ) + 
  geom_point( aes( y=Pred1, colour='Logistic' ), size=2, alpha=0.6 )  +
  geom_abline( slope = 1, intercept = 0, size=2, linetype=2 ) +
  scale_colour_manual( values=c('Logistic'=bp[1]) ) +
  labs( x='Observed', y='Predicted', title='Assessment of NN Model (Training)' )

scatter1

## Histogram of residuals
hist1 <- ggplot( data=myDF ) +
  geom_histogram( aes( x=Resid1, fill='Logistic' ), alpha=0.6, bins = 10 ) +
  scale_fill_manual( values=c('Logistic'=bp[1]) ) +
  labs( x='Residual', y='Frequency', title='Histogram of Residuals of NN Model (Training)')

hist1

## Second NN  ----
# Note the command is the same as the first, only with the activation function changed.
nn2 <- trainNN( trainData=trainData, nnFormula=myFormula, hiddenVec=c(5,3), nnSeed=123, act.fct='tanh' )

scaledPred2 <- compute( nn2, trainData[,xNames] )$net.result
y2 <- unscalePredictions( scaledPred2, unCentre = c1, unScale = c2 )

# Add the info from the second NN to the stored Data Frame
myDF$Pred2 <- y2
myDF$Resid2 <- myDF$Pred2 - myDF$Values

## Scatter plot, including second graph
scatter2 <- ggplot( data=myDF, aes( x=Values ) ) + 
  geom_point( aes( y=Pred1, colour='Logistic' ), size=2, alpha=0.6 )  +
  geom_point( aes( y=Pred2, colour='Tanh' ), size=2, alpha=0.6 )  +
  geom_abline( slope = 1, intercept = 0, size=2, linetype=2 ) +
  scale_colour_manual( values=c('Logistic'=bp[1], 'Tanh'=bp[2]) ) +
  labs( x='Observed', y='Predicted', title='Assessment of NN Model (Training)' )

scatter2

## Histogram of residuals
hist2 <- ggplot( data=myDF ) +
  geom_histogram( aes( x=Resid1, fill='Logistic' ), bins = 10, alpha=0.6, position='identity' ) +
  geom_histogram( aes( x=Resid2, fill='Tanh' ), bins = 10, alpha=0.6, position='identity' ) +
  scale_fill_manual( values=c('Logistic'=bp[1], 'Tanh'=bp[2]) ) +
  labs( x='Residual', y='Frequency', title='Histogram of Residuals of NN Model (Training)')

hist2

## Take the average ----
myDF$Pred3 <- (myDF$Pred1 + myDF$Pred2)/2
myDF$Resid3 <- myDF$Pred3 - myDF$Values

scatter3 <- ggplot( data=myDF, aes( x=Values ) ) + 
  geom_point( aes( y=Pred1, colour='Logistic' ), size=2, alpha=0.6 )  +
  geom_point( aes( y=Pred2, colour='Tanh' ), size=2, alpha=0.6 )  +
  geom_point( aes( y=Pred3, colour='Average' ), size=2, alpha=0.6 )  +
  geom_abline( slope = 1, intercept = 0, size=2, linetype=2 ) +
  scale_colour_manual( values=c('Logistic'=bp[1], 'Tanh'=bp[2], 'Average'=bp[3] ) ) +
  labs( x='Observed', y='Predicted', title='Assessment of NN Model (Training)' )

scatter3

## Histogram of residuals
hist3 <- ggplot( data=myDF ) +
  geom_histogram( aes( x=Resid1, fill='Logistic' ), bins = 10, alpha=0.6, position='identity' ) +
  geom_histogram( aes( x=Resid2, fill='Tanh' ), bins = 10, alpha=0.6, position='identity' ) +
  geom_histogram( aes( x=Resid3, fill='Average' ), bins = 10, alpha=0.6, position='identity' ) +
  scale_fill_manual( values=c('Logistic'=bp[1], 'Tanh'=bp[2], 'Average'=bp[3]) ) +
  labs( x='Residual', y='Frequency', title='Histogram of Residuals of NN Model (Training)')

hist3

## Magic selection of the first two ----
myDF$Pred4 <- with( myDF # Less typing....
                    , ifelse( abs( Resid1 ) < abs( Resid2 )
                      , Pred1 # First prediction is better
                      , Pred2 )
)

myDF$Resid4 <- myDF$Pred4 - myDF$Values

scatter4 <- ggplot( data=myDF, aes( x=Values ) ) + 
  geom_point( aes( y=Pred1, colour='Logistic' ), size=2, alpha=0.6 )  +
  geom_point( aes( y=Pred2, colour='Tanh' ), size=2, alpha=0.6 )  +
  geom_point( aes( y=Pred3, colour='Average' ), size=2, alpha=0.6 )  +
  geom_point( aes( y=Pred4, colour='Selected' ), size=2, alpha=0.6 )  +
  geom_abline( slope = 1, intercept = 0, size=2, linetype=2 ) +
  scale_colour_manual( values=c('Logistic'=bp[1], 'Tanh'=bp[2], 'Average'=bp[3], 'Selected'=bp[4] ) ) +
  labs( x='Observed', y='Predicted', title='Assessment of NN Model (Training)' )

scatter4

## Histogram of residuals
hist4 <- ggplot( data=myDF ) +
  geom_histogram( aes( x=Resid1, fill='Logistic' ), bins = 10, alpha=0.6, position='identity' ) +
  geom_histogram( aes( x=Resid2, fill='Tanh' ), bins = 10, alpha=0.6, position='identity' ) +
  geom_histogram( aes( x=Resid3, fill='Average' ), bins = 10, alpha=0.6, position='identity' ) +
  geom_histogram( aes( x=Resid4, fill='Selected' ), bins = 10, alpha=0.6, position='identity' ) +
  scale_fill_manual( values=c('Logistic'=bp[1], 'Tanh'=bp[2], 'Average'=bp[3], 'Selected'=bp[4]) ) +
  labs( x='Residual', y='Frequency', title='Histogram of Residuals of NN Model (Training)')

hist4

## Selected only to clean up the plots a bit ----
scatterSelected <-ggplot( data=myDF, aes( x=Values ) ) +
  geom_point( aes( y=Pred4, colour='Selected' ), size=2, alpha=0.6 )  +
  geom_abline( slope = 1, intercept = 0, size=2, linetype=2 ) +
  scale_colour_manual( values=c('Logistic'=bp[1], 'Tanh'=bp[2], 'Average'=bp[3], 'Selected'=bp[4] ) ) +
  labs( x='Observed', y='Predicted', title='Assessment of NN Model (Training)' )

scatterSelected

histSelected <- ggplot( data=myDF ) +
  geom_histogram( aes( x=Resid4, fill='Selected' ), bins = 10, alpha=0.6, position='identity' ) +
  scale_fill_manual( values=c('Selected'=bp[4]) ) +
  labs( x='Residual', y='Frequency', title='Histogram of Residuals of NN Model (Training)')

histSelected
