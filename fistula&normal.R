# 加载必要的包
library(dplyr)
library(caret)
library(MASS)
library(pROC)
library(openxlsx)
library(table1)
library(glmnet)
set.seed(1234)
pvalue <- function(x, ...) {
  # Construct vectors of data y, and groups (strata) g
  y <- unlist(x)
  g <- factor(rep(1:length(x), times = sapply(x, length)))
  if (is.numeric(y)) {
    # For numeric variables, perform a standard 2-sample t-test
    p <- t.test(y ~ g)$p.value
  } else {
    # For categorical variables, perform a chi-squared test of independence
    p <- chisq.test(table(y, g))$p.value
  }
  # Format the p-testue, using an HTML entity for the less-than sign.
  # The initial empty string places the output on the line below the variable label.
  c("", sub("<", "&lt;", format.pval(p, digits = 3, eps = 0.001)))
}
data <- read.xlsx('fistula&normal lung radiomics.xlsx')


data_clean <- data %>%
  filter(!is.na(category))  # 去掉fistula缺失的行


data_clean$category <- as.factor(data_clean$category)


model_data <- data_clean[,-c(1,3:39)]
colnames(model_data) <- gsub("-", "_", colnames(model_data))

f1 = glmnet(as.matrix(model_data[,-1]),as.matrix(model_data[,1]),family = "binomial",nlambda = 1000,alpha = 1)
cvfit = cv.glmnet(as.matrix(data.frame(lapply(model_data[,-1],as.numeric))),as.matrix(model_data[,1]),family = "binomial",nlambda = 1000,alpha = 1)

s1 = cvfit$lambda.min
s2 = cvfit$lambda.1se
l.coef1 <- coef(cvfit$glmnet.fit,s = s1,exact = F)
name.lcoef1 = row.names(l.coef1)
x1 = name.lcoef1[which(l.coef1[, 1] != 0)][-1]
x1
j = x1
mod <- j[1]
for (k in j[-1]) {
  mod <- paste(mod, k , sep = "+")
}
mod
fl<- as.formula(paste("~",mod,"| category"))
model_rad <- data.frame(lapply(model_data,as.numeric))
summary_table_lungrad <- table1(
  fl,
  data = model_rad,
  # data = data4,
  overall = F,
  extra.col = list("P-value" = pvalue),
  # render.continuous = custom_format
);summary_table_lungrad



model.formula <- as.formula(paste("category", "~", mod))
model.formula
pred<- predict(cvfit,s = s1,type= "response",newx = as.matrix(data.frame(lapply(model_data[,-1],as.numeric))))
model_data$radscore<- pred[,1]
rad_data <- cbind(data_clean[,1],pred[,1])
rad_data <- data.frame("Patient.Name" =rad_data[,1],"lung_radscore" = rad_data[,2] )
data_clinic <- read.xlsx('patient_names.xlsx')

new_data <- merge(data_clinic,rad_data,by = "Patient.Name")



# data_clean = new_data
# 
# data_clean$Category <- as.factor(data_clean$Category)
# 
# 
# model_data <- data_clean[,-c(1,3:6)]
# summary(model_data)
# # 确保分类变量被转换为因子类型
# # model_data$Bone.hyperplasia_num <- ifelse(model_data$Bone.hyperplasia_num<=2,0,1)
# # model_data$Bone.hyperplasia_num <- as.factor(model_data$Bone.hyperplasia_num)
# model_data$position <- as.factor(model_data$position)
# model_data$Gender <- ifelse(model_data$Gender=="M",1,0)
# model_data$Gender <- as.factor(model_data$Gender)
# model_data$radscore <- as.numeric(model_data$radscore)
# # model_data$radscore <- ifelse(model_data$radscore>0.5,1,0)
# # model_data$radscore <-as.factor(model_data$radscore )
# summary(model_data)
# model_data$Category <- ifelse(model_data$Category=="fistula",1,0)
# model_data$Category <- as.factor(model_data$Category)
# 
# logistic_model <- glm(Category ~ ., 
#                       data = model_data, family = binomial())
# 
# 
# summary(logistic_model)
# 
# 
# predicted_probabilities <- predict(logistic_model, type = "response")
# 
# # 根据预测概率设置阈值进行分类（0.5作为阈值）
# predicted_class <- ifelse(predicted_probabilities > 0.5, 1, 0)
# predicted_class <- as.factor(predicted_class)
# 
# # 混淆矩阵：计算准确率、敏感性、特异性等指标
# confusion_matrix <- confusionMatrix(predicted_class, model_data$Category)
# print(confusion_matrix)
# 
# # 画出ROC曲线评估模型性能
# roc_curve <- roc(model_data$Category, predicted_probabilities)
# plot(roc_curve, main = "ROC Curve", col = "blue")
# 
# # 计算AUC（曲线下面积），评估模型性能
# auc(roc_curve)
# 
# summary_table_surgery0 <- table1(
#   ~ . | Category,
#   data = model_data,
#   # data = data4,
#   overall = F,
#   extra.col = list("P-value" = pvalue),
#   # render.continuous = custom_format
# );summary_table_surgery0

################################################################################
data_spine <- read.xlsx('fistula&normal spine radiomics.xlsx')


data_spine <- data_spine %>%
  filter(!is.na(category))  # 去掉fistula缺失的行


data_spine$category <- as.factor(data_spine$category)


model_data1 <- data_spine[,-c(1,3:39)]
colnames(model_data1) <- gsub("-", "_", colnames(model_data1))
f1 = glmnet(as.matrix(model_data1[,-1]),as.matrix(model_data1[,1]),family = "binomial",nlambda = 1000,alpha = 1)
cvfit = cv.glmnet(as.matrix(data.frame(lapply(model_data1[,-1],as.numeric))),as.matrix(model_data1[,1]),family = "binomial",nlambda = 1000,alpha = 1)

s1 = cvfit$lambda.min
s2 = cvfit$lambda.1se
l.coef1 <- coef(cvfit$glmnet.fit,s = s1,exact = F)
name.lcoef1 = row.names(l.coef1)
x1 = name.lcoef1[which(l.coef1[, 1] != 0)][-1]
x1
j = x1
mod <- j[1]
for (k in j[-1]) {
  mod <- paste(mod, k , sep = "+")
}
mod
fl<- as.formula(paste("~",mod,"| category"))
model_rad1 <- data.frame(lapply(model_data1,as.numeric))
summary_table_spinerad <- table1(
  fl,
  data = model_rad1,
  # data = data4,
  overall = F,
  extra.col = list("P-value" = pvalue),
  # render.continuous = custom_format
);summary_table_spinerad



model.formula <- as.formula(paste("category", "~", mod))
model.formula
pred1<- predict(cvfit,s = s1,type= "response",newx = as.matrix(data.frame(lapply(model_data1[,-1],as.numeric))))
model_data1$radscore<- pred1[,1]
rad_spine <- cbind(data_spine[,1],pred1[,1])
rad_spine <- data.frame("Patient.Name" =rad_spine[,1],"spine_radscore" = rad_spine[,2] )
# data_clinic <- read.xlsx('patient_names.xlsx')

new_data1 <- merge(new_data,rad_spine,by = "Patient.Name")

#######################################################################################
data_media <- read.xlsx('fistula&normal mediastinum radiomics.xlsx')


data_media <- data_media %>%
  filter(!is.na(category))  # 去掉fistula缺失的行


data_media$category <- as.factor(data_media$category)


model_data2 <- data_media[,-c(1,3:39)]
colnames(model_data2) <- gsub("-", "_", colnames(model_data2))
f1 = glmnet(as.matrix(model_data2[,-1]),as.matrix(model_data2[,1]),family = "binomial",nlambda = 1000,alpha = 1)
cvfit = cv.glmnet(as.matrix(data.frame(lapply(model_data2[,-1],as.numeric))),as.matrix(model_data2[,1]),family = "binomial",nlambda = 1000,alpha = 1)

s1 = cvfit$lambda.min
s2 = cvfit$lambda.1se
l.coef1 <- coef(cvfit$glmnet.fit,s = s1,exact = F)
name.lcoef1 = row.names(l.coef1)
x1 = name.lcoef1[which(l.coef1[, 1] != 0)][-1]
x1
j = x1
mod <- j[1]
for (k in j[-1]) {
  mod <- paste(mod, k , sep = "+")
}
mod
fl<- as.formula(paste("~",mod,"| category"))
model_rad2 <- data.frame(lapply(model_data2,as.numeric))
summary_table_esophagusrad <- table1(
  fl,
  data = model_rad2,
  # data = data4,
  overall = F,
  extra.col = list("P-value" = pvalue),
  # render.continuous = custom_format
);summary_table_esophagusrad

model.formula <- as.formula(paste("category", "~", mod))
model.formula
pred2<- predict(cvfit,s = s1,type= "response",newx = as.matrix(data.frame(lapply(model_data2[,-1],as.numeric))))
model_data2$radscore<- pred2[,1]
rad_media <- cbind(data_media[,1],pred2[,1])
rad_media <- data.frame("Patient.Name" =rad_media[,1],"media_radscore" = rad_media[,2] )

new_data2 <- merge(new_data1,rad_media,by = "Patient.Name")







data_clean = new_data2

data_clean$Category <- as.factor(data_clean$Category)


model_data <- data_clean[,-c(1,3:6,9,10)]
summary(model_data)
# 确保分类变量被转换为因子类型
model_data$Bone.hyperplasia_num <- ifelse(model_data$Bone.hyperplasia_num<=2,0,1)
model_data$Bone.hyperplasia_num <- as.factor(model_data$Bone.hyperplasia_num)
model_data$position <- as.factor(model_data$position)
model_data$Gender <- ifelse(model_data$Gender=="M",1,0)
model_data$Gender <- as.factor(model_data$Gender)
model_data$lung_radscore <- as.numeric(model_data$lung_radscore)
model_data$spine_radscore <- as.numeric(model_data$spine_radscore)
model_data$media_radscore <- as.numeric(model_data$media_radscore)
# model_data$spine_radscore <- ifelse(model_data$spine_radscore>mean(model_data$spine_radscore),1,0)
# model_data$spine_radscore<-as.factor(model_data$spine_radscore)
# model_data$media_radscore <- ifelse(model_data$media_radscore>mean(model_data$media_radscore),1,0)
# model_data$media_radscore<-as.factor(model_data$media_radscore)
# model_data$lung_radscore <- ifelse(model_data$lung_radscore>mean(model_data$lung_radscore),1,0)
# model_data$lung_radscore<-as.factor(model_data$lung_radscore)
summary(model_data)
model_data$Category <- ifelse(model_data$Category=="fistula",1,0)
model_table <- model_data[,1:8]
summary_table_surgery0 <- table1(
  ~ . | Category,
  data = model_table,
  # data = data4,
  overall = F,
  extra.col = list("P-value" = pvalue),
  # render.continuous = custom_format
);summary_table_surgery0


model_data$Category <- as.factor(model_data$Category)

logistic_model <- glm(Category ~ ., 
                      data = model_data, family = binomial())


summary(logistic_model)


predicted_probabilities <- predict(logistic_model, type = "response")

# 根据预测概率设置阈值进行分类（0.5作为阈值）
predicted_class <- ifelse(predicted_probabilities > 0.5, 1, 0)
predicted_class <- as.factor(predicted_class)

# 混淆矩阵：计算准确率、敏感性、特异性等指标
confusion_matrix <- confusionMatrix(predicted_class, model_data$Category)
print(confusion_matrix)

# 画出ROC曲线评估模型性能
roc_curve <- roc(model_data$Category, predicted_probabilities)
plot(roc_curve, main = "ROC Curve", col = "blue")

# 计算AUC（曲线下面积），评估模型性能
auc(roc_curve)

summary_table_surgery0 <- table1(
  ~ . | Category,
  data = model_data,
  # data = data4,
  overall = F,
  extra.col = list("P-value" = pvalue),
  # render.continuous = custom_format
);summary_table_surgery0




#############################################################################################

# 假设 model_data 是你的数据集，Category 是因变量
univariate_models <- list()

# 获取所有自变量的名称
# predictors <- names(model_data)[names(model_data) != "Category"]
predictors <- names(model_data)[names(model_data) != c("Category","lung_radscore","spine_radscore", "media_radscore")]
# 对每个自变量进行单因素逻辑回归分析
for (predictor in predictors) {
  formula <- as.formula(paste("Category ~", predictor))
  univariate_model <- glm(formula, data = model_data, family = binomial())
  univariate_models[[predictor]] <- summary(univariate_model)
}

# 打印每个单因素逻辑回归模型的结果
for (predictor in predictors) {
  cat("Univariate logistic regression for:", predictor, "\n")
  print(univariate_models[[predictor]])
  cat("\n")
}





library(forestplot)
# 提取多因素逻辑回归模型的结果
# 提取多因素逻辑回归模型的结果
model_summary <- summary(logistic_model)

# 提取系数、标准误、置信区间和 p 值
coefficients <- coef(logistic_model)
conf_intervals <- confint(logistic_model)
p_values <- model_summary$coefficients[, 4]

# 将结果整理为数据框
results <- data.frame(
  Predictor = rownames(model_summary$coefficients),
  OR = exp(coefficients),  # 计算 odds ratio (OR)
  LowerCI = exp(conf_intervals[, 1]),  # 置信区间下限
  UpperCI = exp(conf_intervals[, 2]),  # 置信区间上限
  PValue = p_values
)

# 去掉截距项（通常不需要在森林图中显示）
results <- results[-c(1,10:12), ]
# 定义森林图的标签
labeltext <- cbind(
  c("Predictor", results$Predictor),  # 自变量名称
  c("OR", round(results$OR, 2)),      # OR 值
  c("95% CI", paste0("(", round(results$LowerCI, 2), " - ", round(results$UpperCI, 2), ")")),  # 置信区间
  c("P Value", round(results$PValue, 3))  # P 值
)

# 绘制森林图
forestplot(
  labeltext = labeltext,
  mean = c(NA, results$OR),  # OR 值
  lower = c(NA, results$LowerCI),  # 置信区间下限
  upper = c(NA, results$UpperCI),  # 置信区间上限
  zero = 1,  # OR = 1 的参考线
  boxsize = 0.2,  # 点的大小
  lineheight = "auto",  # 行高
  col = fpColors(box = "royalblue", line = "darkblue"),  # 颜色
  xlab = "Odds Ratio (95% CI)",  # X 轴标签
  txt_gp = fpTxtGp(cex = 0.8)  # 文字大小
)
