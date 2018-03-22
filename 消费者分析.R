if(!suppressWarnings(require("mice"))){
  install.packages("mice")
  require("mice")
}
if(!suppressWarnings(require("VIM"))){
  install.packages("VIM")
  require("VIM")
}
if(!suppressWarnings(require("DMwR"))){
  install.packages("DMwR")
  require("DMwR")
}
if(!suppressWarnings(require("psych"))){
  install.packages("psych")
  require("psych")
}
if(!suppressWarnings(require("ggplot2"))){
  install.packages("ggplot2")
  require("ggplot2")
}
if(!suppressWarnings(require("MASS"))){
  install.packages("MASS")
  require("MASS")
}

if(!suppressWarnings(require("caret"))){
  install.packages("caret")
  require("caret")
}
if(!suppressWarnings(require("randomForest"))){
  install.packages("randomForest")
  require("randomForest")
}
if(!suppressWarnings(require("RSNNS"))){
  install.packages("RSNNS")
  require("RSNNS")
}
setwd("C:/Users/Zoey/Desktop")
yp<-read.csv("ypr.csv",header=T)
#####################################缺失值处理    #####################################
sum(is.na(yp))
miss<-md.pattern(yp)
write.csv(miss,"miss.csv", row.names=T)
manyNAs(yp)#统计缺失超过20%的样本
pMiss <- function(x){sum(is.na(x))/length(x)}
sum(apply(yp,2,pMiss)>0.2)#统计缺失超过20%的的变量
cdata<-na.omit(yp)#删除缺失样本
x<-as.data.frame(abs(is.na(yp)))
nac<-x[which(colMeans(x)>0)]#含有缺失值的变量
#k近邻插补
kyp<-knnImputation(yp, k = 10, scale = T, meth = "weighAvg", distData = NULL)
sum(is.na(kyp))
kyp<-round(kyp)
write.csv(kyp,"kyp.csv", row.names=T)

#####################################lof异常值（数值型数据）#####################################
# k是计算局部异常因子所需要判断异常点周围的点的个数
outlier.scores <- lofactor(kyp, k=5)
# 绘制异常值得分的密度分布图
plot(density(outlier.scores))
# 挑出得分排前5%的数据作为异常值
outliers <- order(outlier.scores, decreasing = T)[1:50]
# 输出异常值
a<-print(outliers)

######################################聚类检验异常值####################################
kmeans.result <- kmeans(kyp, centers=3)
# 输出簇中心
kmeans.result$centers
# 分类结果
kmeans.result$cluster
# 计算数据对象与簇中心的距离
centers <- kmeans.result$centers[kmeans.result$cluster, ]
distances <- sqrt(rowSums((kyp - centers)^2))
# 挑选出前5%最大距离的样本
outliers1 <- order(distances, decreasing = T)[1:50]
# 输出异常值
plot(1:900,distances,xlim=c(0,900),xlab="sample",ylab="distance") 
b<-print(outliers1)
######################################合并取离群点####################################
c<-intersect(a,b) #取交集
alone<-print(kyp[c,])
write.csv(alone,"alone.csv", row.names=T)
#####################################主成分分析#################################
data<-kyp[,134:140]
data<-data[-c(c),]#剔除离群点
d<-kyp[,-c(134:140)]
d<-d[-c(c),]
z1<-principal(data,2,rotate="varimax")
summary(z1)
z1$loadings
z1$scores
#####################################聚类分析#################################  
km=kmeans(data,center=3,algorithm =c("MacQueen"))  
type<-as.character.Date(km$cluster)
qplot(z1$scores[,1],z1$scores[,2],colour=type,xlab = "消费因子",ylab = "节约因子") 
yp.a<-cbind(d,type)#数据合并
#####################################定序回归#####################################
sam<- sample(2,nrow(yp.a),replace = T,prob=c(0.3,0.7))
train <- yp.a[sam==2,]
test<-yp.a[sam==1,]
model1=polr(as.factor(train$a)~.,method="logistic", Hess=T,data=train)
summary(model1)
p1<- predict(model1,test)
#预测 
(preTable<-table(p1,test$a)) 
(accuracy<-sum(diag(preTable))/sum(preTable))#预测精度 
#####################################随机森林 #####################################
Randommodel <- randomForest(yp.a$a~ ., data=yp.a,importance = TRUE, proximity = FALSE, ntree = 300)
print(Randommodel)  

#####################################神经网络##################################
input<-yp.a[,-144]
Targets = decodeClassLabels(yp.a$a)
#从中划分出训练样本和检验样本 
yp = splitForTrainingAndTest(input,Targets,ratio=0.3)
#利用mlp命令执行前馈反向传播神经网络算法 
yp= normTrainingAndTestSet(yp)
model = mlp(yp$inputsTrain,yp$targetsTrain ,size=3, learnFunc="Quickprop", 
              learnFuncParams=c(0.1,0.01,2),maxit=500, 
              inputsTest=yp$inputsTest, targetsTest=yp$targetsTest)  
#利用上面建立的模型进行预测 
predictions = predict(model,yp$inputsTest)
#生成混淆矩阵，观察预测精度 
preTable1<-confusionMatrix(yp$targetsTest,predictions)
(accuracy<-sum(diag(preTable1))/sum(preTable1))



