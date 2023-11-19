from aaindex.aaindex1 import aaindex1
import aaindex



#AAindex1 为20个数值的氨基酸指数
#AAindex2:氨基酸突变矩阵
#AAindex3:统计蛋白质接触电位
full_record  =  aaindex1 [ 'CHOP780206' ]    #get full AAI record
''' 以上语句将返回 -> 
{'description': 'N 端非螺旋区域的归一化频率 (Chou-Fasman, 1978b)', 'notes': '', 
'refs'：“Chou, PY 和 Fasman, GD '从氨基酸序列预测蛋白质的二级结构'Adv. Enzymol. 47, 45-148 (1978); Kawashima, S. 和 Kanehisa, M 'AAindex：氨基酸索引数据库。Nucleic Acids Res. 28, 374 (2000).", 
'values': {'-': 0, 'A': 0.7, 'C': 0.65, 'D': 0.98, 'E': 1.04, 'F '：0.93，'G'：1.41，'H'：1.22，'I'：0.78，'K'：1。}
'''
#获取 AAIndex 记录的各个元素
record_values=aaindex1 [ 'CHOP780206' ][ 'values' ]
record_description =aaindex1 [ 'CHOP780206' ][ 'description' ]
# record_references = aaindex1 [ 'CHOP780206' ][ 'refs' ]
print(record_values)
print(record_description)
# print(record_references)

print(aaindex1.get_all_categories())
# print(aaindex1.aaindex_json)
print(aaindex1.record_codes()) #记录总数
print(aaindex1.record_names()) #记录名字
print(len(aaindex1.record_names()))
print(aaindex1.amino_acids())