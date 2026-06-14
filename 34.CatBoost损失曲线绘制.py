import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.font_manager import FontProperties

# 设置字体
FONT_PATH = "/home/wangzonghan/bisheshuju/fonts/SimHei.ttf"
my_font = FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else FontProperties()
plt.rcParams['axes.unicode_minus'] = False

# 读取 CatBoost 自动生成的日志文件
learn_error_path = "/home/wangzonghan/1/catboost_info/learn_error.tsv"
test_error_path = "/home/wangzonghan/1/catboost_info/test_error.tsv"

df_learn = pd.read_csv(learn_error_path, sep='\t')
df_test = pd.read_csv(test_error_path, sep='\t')

# 绘图
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(df_learn['iter'], df_learn['RMSE'], label='训练集 (Train RMSE)', color='#1f77b4', linewidth=2)
plt.plot(df_test['iter'], df_test['RMSE'], label='验证集 (Validation RMSE)', color='#ff7f0e', linewidth=2, linestyle='--')

plt.title('CatBoost 模型迭代收敛与损失曲线', fontproperties=my_font, fontsize=16)
plt.xlabel('迭代次数 (Iterations)', fontproperties=my_font, fontsize=14)
plt.ylabel('均方根误差 RMSE (μg/m³)', fontproperties=my_font, fontsize=14)
plt.legend(prop=my_font, fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)

# 保存图片
output_dir = "/home/wangzonghan/bisheshuju/Results/Figures_CatBoost"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "CatBoost_Loss_Curve.png"), bbox_inches='tight')
print("✅ CatBoost 损失曲线已生成！")