

def get_number():


    f2=open('index_number.txt', 'w')
    with open('CandidaAlbicansPos.fasta', mode='r') as f:

        for line in f.readlines():
            if line.startswith('>'):
                print(line.strip())
                # print(line.split()[1])
                f2.write(line.strip()+'\n')

    f.close()
    f2.close()



def get_mapId():

    path= '../Datasets/CandidaAlbicans.fasta'
    f2=open('MapId.txt', mode='w')
    with open(path,mode='r') as f:

        for line in f.readlines():

            if line.startswith('>'):
                t1=line.index('|')
                print(t1)
                t2=line.index('|',t1+1)
                print(t2)
                # print(line)
                print(line[t1+1:t2])
                f2.write(line[t1+1:t2]+'\n')

    f.close()
    f2.close()


def get_length():

    strmap='MQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGGKKRKKKVYTTPKKIKHKHRKHKLAVLTYYKVDNEGNVERLRRECPAPTCGAGIFMANMKDRQYCGKCHLTLKAN'

    index=0
    for i in strmap:
        # print(i)
        index+=1
        if i == 'K':
            print(index)

    # print(len(strmap))
if __name__ == '__main__':

    # get_number()

    # get_mapId()
    get_length()