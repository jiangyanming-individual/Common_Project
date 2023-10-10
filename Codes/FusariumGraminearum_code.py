# record the khib site contained peptide

sitepep = {}
import xlwings as xw
import re

"""
禾谷镰刀菌：
"""
app = xw.App(visible=False, add_book=False)
excle_path = 'D:\Program Files (x86)\Python_workspace\Common_Project\Datasets\FusariumGraminearum.xlsx'

wb = app.books.open(excle_path)
#read sheet
# sht = wb.sheets('Table S2')
sht = wb.sheets('Site_quant')


#记录位点对应的蛋白质ID
for r in range(2, 3503):

    print("r value:",r)
    key = sht[r, 0].value
    print("key value:",key)
    if sitepep.get(key):
        sitepep[key].append(int(sht[r, 1].value))
    else:
        sitepep[key] = [int(sht[r, 1].value)]
wb.close()

fasta_file = 'D:\Program Files (x86)\Python_workspace\Common_Project\Datasets\FusariumGraminearum.fasta'
output_Pos_path = '../Output_file/FusariumGraminearumPos.fasta'
output_Neg_path = '../Output_file/FusariumGraminearumNeg.fasta'


f1 = open(output_Pos_path, 'w')
f2 = open(output_Neg_path, 'w')
f = open(fasta_file)

n = 1
seq = ''
for line in f:
    if (line.startswith('>')):
        if n != 1:
            khib = sitepep[pid]
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
        t1 = line.index('|')
        t2 = line.index('|', t1 + 1)
        pid = line[t1 + 1:t2]
        n = n + 1
        seq = ''

    else:
        seq = seq + line.strip()

khib = sitepep[pid]

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
