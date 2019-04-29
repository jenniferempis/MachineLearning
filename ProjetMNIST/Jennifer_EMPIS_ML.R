## Machine learning

## Il faut tout d'abord charger les donnees appelees mnist_train.csv

## path a modifier en fonction du lieu du fichier
path="~/Bureau/stockage/Cours/S7/MachineLearning/Ressources-20190108/mnist_train.csv/"
data <- read.csv(paste(path,"mnist_train.csv"), header=FALSE);data
print(table(data$V1)) # repartition des y

## On procede ensuite en scindant de maniere aleatoire les donnees contenues dans Boston en 2 sous-ensembles la base d'entrainement "train" et la base de test "test".  

set.seed(123)
index <- sample(1:nrow(data),round(0.75*nrow(data)))
train <- data[index,]
test <- data[-index,]
V1=data[,1]
# Nous mettons donc a l'echelle et scindons les donnees avant de poursuivre:
normalized <- (data[,-1]-min(data[,-1]))/(max(data[,-1])-min(data[,-1]))
scaled <- cbind(V1,normalized)
is.data.frame(scaled) # TRUE

train_ <- scaled[index,]
test_ <- scaled[-index,]

# Nous ajustons ensuite un modÃ¨le de regression lineaire et le testons sur la base de test.
# Notez qu'ici est utilisee la fonction gml () au lieu de lm ().
# Cela deviendra utile plus tard lors de la validation croisee du modele lineaire.

lm.fit <- glm(V1~., data=train)
summary(lm.fit)
pr.lm <- predict(lm.fit,test)
summary(pr.lm)
MSE.lm <- sum((pr.lm - test$V1)^2)/nrow(test)
summary(MSE.lm)


## 1E METHODE AVEC NEURALNET

library(nnet)
library(neuralnet)

n <- names(train_)
f <- (paste("V1 ~", paste(n[!n %in% "V1"], collapse = " + ")))

f <- as.formula(paste("V1 ~", paste(n[!n %in% "V1"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(256,128,64,32,10))

# Le paquet neuralnet permet de faire appel a un bon outil pour tracer le modele:

plot.nnet(nn)

# Prédire V1 en utilisant le réseau de neurones
pr.nn<-compute(nn,test_[,-1]);pr.nn
pr.nn2<-pr.nn$net.result*(max(data$V1)-min(data$V1))+min(data$V1);pr.nn2
test_r<-(test_$V1)*(max(data$V1)-min(data$V1))+min(data$V1);test_r
MSE.nn <- sum ((test_r - pr.nn2) ^ 2) / nrow (test_)

# On peut ensuite comparer les 2 MSE
print (paste (MSE.lm, MSE.nn))



## 2E METHODE AVEC KERAS

library(keras)
install_keras()


# Nous commençons par creer un modèle séquentiel, puis en ajoutant des couches.
# Ici on choisi de prendre 5 couches cachees de taille respectives : 256,128,64,32,10.
model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model) # imprime les détails du modèle

# Ensuite, on compile le modele avec la fonction de perte,
# l'optimiseur et les mesures appropries:

sgd <- optimizer_sgd(lr = 0.01) 

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = sgd,
  metrics = c('accuracy')
)

train_y=to_categorical(train_[,1],10)
# Les donnees y sont un vecteur entier avec des valeurs allant de 0 à 9.

# On utilise des lots de 128 images
# On peut faire varier l option epochs 
history <- model %>% fit(
  as.matrix(train_[,-1]),train_y,
  epochs = 40, batch_size = 128,
  validation_split = 0.2
)


plot(history)

test_y=to_categorical(test_[,1],10)
# On evalue les performances du modèle sur les donnees de test
model %>% evaluate(as.matrix(test[,-1]), test_y)

## $acc = 0.9572667
## On obtient un pourcentage de 95% de bonnes classifications
## Le modèle obtenu est donc correct

