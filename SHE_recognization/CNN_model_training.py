from TF_CNN import CNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

CSV_FILE_PATH = 'E://face_data/sample_test.csv'          # CSV 文件路径
IRIS = pd.read_csv(CSV_FILE_PATH)       # 读取CSV文件
target_variable = 'label'               # 目标变量

# 数据集的特征
features = list(IRIS.columns)
features.remove(target_variable)

# 目标变量的类别
Class = IRIS[target_variable].unique()

# 对目标变量进行重新编码
# 目标变量的类别字典
Class_dict = {}
for i, clf in enumerate(Class):
    Class_dict[clf] = i+1
# 增加一列target, 将目标变量进行编码
IRIS['target'] =  IRIS[target_variable].apply(lambda x: Class_dict[x])

# 对目标变量进行0-1编码
lb = LabelBinarizer()
lb.fit(list(Class_dict.values()))
transformed_labels = lb.transform(IRIS['target'])
y_bin_labels = []   # 对多分类进行0-1编码的变量
for i in range(transformed_labels.shape[1]):
    y_bin_labels.append('y'+str(i))
    IRIS['y'+str(i)] = transformed_labels[:, i]
# print(IRIS.head(10))

# 数据是否标准化
# x_bar = (x-mean)/std
IS_STANDARD = 'yes'
if IS_STANDARD == 'yes':
    for feature in features:
        mean = IRIS[feature].mean()
        std = IRIS[feature].std()
        IRIS[feature] = (IRIS[feature]-mean)/std

# 将数据集分为训练集和测试集，训练集70%, 测试集30%
x_train, x_test, y_train, y_test = train_test_split(IRIS[features], IRIS[y_bin_labels], \
                                                    train_size = 0.7, test_size=0.3, random_state=123)

# 使用CNN进行预测
# 构建CNN网络
# 模型保存地址
MODEL_SAVE_PATH = 'E://logs/cnn_she_model.ckpt'
# CNN初始化
cnn = CNN(100, 0.0005, MODEL_SAVE_PATH)

# 训练CNN
cnn.train(x_train, y_train)
# 预测数据
y_pred = cnn.predict(x_test)

# 预测分类
prediction = []
for pred in y_pred:
    prediction.append(list(pred).index(max(pred))+1)

# 计算预测的准确率
x_test['prediction'] = prediction
x_test['target'] = IRIS['target'][y_test.index]
print(x_test.head(n=20))
accuracy = accuracy_score(x_test['prediction'], x_test['target'])
print('CNN的预测准确率为%.2f%%.'%(accuracy*100))