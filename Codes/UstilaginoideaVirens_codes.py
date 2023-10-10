# record the khib site contained peptide

sitepep = {}
replacePep={}
import xlwings as xw
import re

"""
尖孢镰刀菌
"""
app = xw.App(visible=False, add_book=False)
excle_path = 'D:\Program Files (x86)\Python_workspace\Common_Project\Datasets\\Ustilaginoidea_virens.xlsx'

wb = app.books.open(excle_path)
#read sheet
# sht = wb.sheets('Table S2')
sht = wb.sheets('Ustilaginoidea virens')


def read_file():

    index_list=[]
    with open('../back/index.txt', 'r') as f:
        for line in f.readlines():
            index_list.append(line.strip().strip('\n'))
    # print(index_list)
    return index_list


repalce_list=read_file()
#记录位点对应的蛋白质ID
for r in range(3, 3429):

    print("r value:",r)
    key = sht[r, 0].value
    print("key value:",key)
    if sitepep.get(key):
        sitepep[key].append(int(sht[r, 1].value))
    else:
        sitepep[key] = [int(sht[r, 1].value)]
wb.close()
print("sitepep:", sitepep.items())

# repalce 替换
i = 0
for item in sitepep.items():
    # print(item)
    replacePep[repalce_list[i]] = item[1]  # key:value
    # print(replacePep.items())
    i += 1

print("replacePep:", replacePep.items())
# print("i:", i)


fasta_file = 'D:\Program Files (x86)\Python_workspace\Common_Project\Datasets\\Ustilaginoidea_virens.fasta'
output_Pos_path = '../Output_file/UstilaginoideaVirenPos.fasta'
output_Neg_path = '../Output_file/UstilaginoideaVirenNeg.fasta'


f1 = open(output_Pos_path, 'w')
f2 = open(output_Neg_path, 'w')
f = open(fasta_file)

n = 1
seq = ''


for line in f:
    if (line.startswith('>')):
        if n != 1:
            khib = replacePep[pid]
            p = seq.find('K')
            while (p != -1):
                if p - 18 < 0:
                    peptide = 'X' * (18 - p) + seq[0:p + 19]
                else:
                    peptide = seq[p - 18:p + 19]
                if len(seq) - (p + 1) < 18:
                    peptide += 'X' * (p + 19 - len(seq))
                if p + 1 in khib:
                    f1.write('>{}\t{}\n{}\n'.format(pid, p + 1, peptide))
                else:
                    f2.write('>{}\t{}\n{}\n'.format(pid, p + 1, peptide))
                p = seq.find('K', p + 1)

        t1 = line.index('>')
        t2 = line.index(' ')
        pid = line[t1 + 1:t2]
        n = n + 1
        seq = ''

    else:
        seq = seq + line.strip()

khib = replacePep[pid]

p = seq.find('K')
while (p != -1):
    if p - 18 < 0:
        peptide = 'X' * (18 - p) + seq[0:p + 19]
    else:
        peptide = seq[p - 18:p + 19]
    if len(seq) - (p + 1) < 18:
        peptide += 'X' * (p + 19 - len(seq))
    if p + 1 in khib:
        f1.write('>{}\t{}\n{}\n'.format(pid, p + 1, peptide))
    else:
        f2.write('>{}\t{}\n{}\n'.format(pid, p + 1, peptide))
    p = seq.find('K', p + 1)

f1.close()
f2.close()
f.close()
print("处理完成！")
