

def get_index():
    fasta_file = '../Datasets/Ustilaginoidea_virens.fasta'
    # output_Pos_path = '../Output_file/UstilaginoideaVirenPos.fasta'
    # output_Neg_path = '../Output_file/UstilaginoideaVirenNeg.fasta'

    output_file = 'index.txt'

    f1 = open(output_file, 'w', )
    f = open(fasta_file, mode='r', encoding='utf-8')

    n = 0
    seq = ''
    for line in f:
        if (line.startswith('>')):
            # print(line.index('>'))
            t1 = line.index('>')
            t2 = line.index(' ')
            print(line[t1 + 1:t2])
            n += 1
            # print(line.strip().split(' '))

            f1.write(line[t1 + 1:t2] + '\n')

    print("n:", n)
    f1.close()
    f.close()
    print("处理完成！")



def read_file():

    index_list=[]
    with open('index.txt', 'r') as f:
        for line in f.readlines():
            index_list.append(line.strip().strip('\n'))
    # print(index_list)
    return index_list

def get_excel_index():

    sitepep = {}

    replacePep={}
    import xlwings as xw
    import re

    """
    尖孢镰刀菌
    """
    app = xw.App(visible=False, add_book=False)
    excle_path = "/Datasets/Ustilaginoidea_virens.xlsx"

    wb = app.books.open(excle_path)
    #read sheet
    # sht = wb.sheets('Table S2')
    sht = wb.sheets('Ustilaginoidea virens')

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
    print("sitepep:",sitepep.items())

    i=0
    for item in sitepep.items():
        # print(item)
        replacePep[repalce_list[i]]=item[1] # key:value
        # print(replacePep.items())
        i+=1

    print("replacePep:",replacePep.items())
    print("i:",i)

if __name__ == '__main__':

    # get_index()
    get_excel_index()



