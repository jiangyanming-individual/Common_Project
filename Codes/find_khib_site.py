#id map
idmap={}
f=open('data/rice-seeds.tab')
for line in f:
    col=line.split()
    if not ',' in col[0]:
        idmap[col[0]]=col[1]
    else:
        q1,q2=col[0].split(',')
        idmap[q1]=col[1]
        idmap[q2]=col[1]
f.close()

#record the khib site contained peptide
sitepep={}
import xlwings as xw
app=xw.App(visible=False,add_book=False)
wb=app.books.open('data/rice-seeds.xlsx')
sht=wb.sheets('Table S1')
for r in range(14,9930):
    pid=sht[r,0].value
    seq=sht[r,1].value
    key=idmap[pid]
    if sitepep.get(key):
        sitepep[key].append(seq)
    else:
        sitepep[key]=[seq]
wb.close()

import re
f1=open('peptide/rice-seeds-pos','w')
f2=open('peptide/rice-seeds-neg','w')
f=open('data/rice-seeds.fasta')
n=1;
seq=''
for line in f:
    if(line.startswith('>')):
        if n!=1:
            if pid in sitepep:
                khib=[]
                for pep in sitepep[pid]:
                    pep=pep.strip('_')
                    pep=pep.replace('K(hib)','k')
                    pep=re.sub('\(.*?\)','',pep)
                    p=pep.find('k')
                    pep=pep.replace('k','K')
                    if seq.find(pep)!=-1:
                        khib.append(seq.index(pep)+p)
                p=seq.find('K')
                while(p!=-1):
                        if p-15<0:
                            peptide='X' * (15-p) + seq[0:p+16]
                        else:
                            peptide=seq[p-15:p+16]
                        if len(seq)-(p+1)<15:
                            peptide+= 'X'*(p+16-len(seq))
                        if p in khib:
                            f1.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
                        else:
                            f2.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
                        p=seq.find('K',p+1)
        t1=line.index('|')
        t2=line.index('|',t1+1)
        pid=line[t1+1:t2]
        n=n+1
        seq=''
        
    else:
        seq=seq+line.strip()

khib=[]
for pep in sitepep[pid]:
    pep=pep.strip('_')
    pep=pep.replace('K(hib)','k')
    pep=re.sub('\(.*?\)','',pep)
    p=pep.find('k')
    pep=pep.replace('k','K')
    khib.append(seq.index(pep)+p)
p=seq.find('K')
while(p!=-1):
        if p-15<0:
            peptide='X' * (15-p) + seq[0:p+16]
        else:
            peptide=seq[p-15:p+16]
        if len(seq)-(p+1)<15:
            peptide+= 'X'*(p+16-len(seq))
        if p in khib:
            f1.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
        else:
            f2.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
        p=seq.find('K',p+1)
f1.close()
f2.close()
f.close()

idmap={}
f=open('data/rice-leave.tab')
for line in f:
    col=line.split()
    idmap[col[0]]=col[1]
f.close()

#record the khib site contained peptide
sitepep={}
import xlwings as xw
import re
app=xw.App(visible=False,add_book=False)
wb=app.books.open('data/rice-leave.xlsx')
sht=wb.sheets('Annotation_Combine')
for r in range(5,4168):
    pid=sht[r,0].value
    seq=sht[r,8].value
    prob=sht[r,5].value
    if 'K(1)' in seq:
        seq=seq.replace('K(1)','k')
    else:
        seq=seq.replace('K('+str(round(prob,3))+')','k')
        seq=re.sub('\(.*?\)','',seq)
    key=idmap[pid]
    if sitepep.get(key):
        sitepep[key].append(seq)
    else:
        sitepep[key]=[seq]
wb.close()

f1=open('rice-leave-pos','w')
f2=open('rice-leave-neg','w')
f=open('data/rice-leave.fasta')
n=1;
seq=''
for line in f:
    if(line.startswith('>')):
        if n!=1:
            khib=[]
            for pep in sitepep[pid]:
                p=pep.find('k')
                pep=pep.replace('k','K')
                khib.append(seq.index(pep)+p)
            p=seq.find('K')
            while(p!=-1):
                if p-15<0:
                    peptide='X' * (15-p) + seq[0:p+16]
                else:
                    peptide=seq[p-15:p+16]
                if len(seq)-(p+1)<15:
                    peptide+= 'X'*(p+16-len(seq))
                if p in khib:
                    f1.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
                else:
                    f2.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
                p=seq.find('K',p+1)
        t1=line.index('|')
        t2=line.index('|',t1+1)
        pid=line[t1+1:t2]
        n=n+1
        seq=''
        
    else:
        seq=seq+line.strip()

khib=[]
for pep in sitepep[pid]:
    p=pep.find('k')
    pep=pep.replace('k','K')
    khib.append(seq.index(pep)+p)
p=seq.find('K')
while(p!=-1):
        if p-15<0:
            peptide='X' * (15-p) + seq[0:p+16]
        else:
            peptide=seq[p-15:p+16]
        if len(seq)-(p+1)<15:
            peptide+= 'X'*(p+16-len(seq))
        if p in khib:
            f1.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
        else:
            f2.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
        p=seq.find('K',p+1)
f1.close()
f2.close()
f.close()

idmap={}
f=open('data/wheat-leave.tab')
for line in f:
    col=line.split()
    idmap[col[0]]=col[1]
f.close()

#record the khib site contained peptide
sitepep={}
import xlwings as xw
import re
app=xw.App(visible=False,add_book=False)
wb=app.books.open('data/wheat-leave.xlsx')
sht=wb.sheets('S1')
for r in range(1,6043):
    pid=sht[r,0].value
    seq=sht[r,3].value
    prob=sht[r,5].value
    if 'K(1)' in seq:
        seq=seq.replace('K(1)','k')
    else:
        seq=seq.replace('K('+str(round(prob,3))+')','k')
        seq=re.sub('\(.*?\)','',seq)
    key=idmap[pid]
    if sitepep.get(key):
        sitepep[key].append(seq)
    else:
        sitepep[key]=[seq]
wb.close()

f1=open('rice-leave-pos','w')
f2=open('rice-leave-neg','w')
f=open('data/rice-leave.fasta')
n=1;
seq=''
for line in f:
    if(line.startswith('>')):
        if n!=1:
            khib=[]
            for pep in sitepep[pid]:
                p=pep.find('k')
                pep=pep.replace('k','K')
                khib.append(seq.index(pep)+p)
            p=seq.find('K')
            while(p!=-1):
                if p-15<0:
                    peptide='X' * (15-p) + seq[0:p+16]
                else:
                    peptide=seq[p-15:p+16]
                if len(seq)-(p+1)<15:
                    peptide+= 'X'*(p+16-len(seq))
                if p in khib:
                    f1.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
                else:
                    f2.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
                p=seq.find('K',p+1)
        t1=line.index('|')
        t2=line.index('|',t1+1)
        pid=line[t1+1:t2]
        n=n+1
        seq=''
        
    else:
        seq=seq+line.strip()

khib=[]
for pep in sitepep[pid]:
    p=pep.find('k')
    pep=pep.replace('k','K')
    khib.append(seq.index(pep)+p)
p=seq.find('K')
while(p!=-1):
        if p-15<0:
            peptide='X' * (15-p) + seq[0:p+16]
        else:
            peptide=seq[p-15:p+16]
        if len(seq)-(p+1)<15:
            peptide+= 'X'*(p+16-len(seq))
        if p in khib:
            f1.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
        else:
            f2.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
        p=seq.find('K',p+1)
f1.close()
f2.close()
f.close()

#record the khib site contained peptide
sitepep={}
import xlwings as xw
import re
app=xw.App(visible=False,add_book=False)
wb=app.books.open('data/soybean-leave.xlsx')
sht=wb.sheets('Overlap')
for r in range(2,4253):
    key=sht[r,0].value
    if sitepep.get(key):
        sitepep[key].append(int(sht[r,1].value))
    else:
        sitepep[key]=[int(sht[r,1].value)]
wb.close()

f1=open('soybeanLeavePos','w')
f2=open('soybeanLeaveNeg','w')
f=open('data/soybean.fasta')
n=1;
seq=''
for line in f:
    if(line.startswith('>')):
        if n!=1:
            khib=sitepep[pid]
            p=seq.find('K')
            while(p!=-1):
                if p-15<0:
                    peptide='X' * (15-p) + seq[0:p+16]
                else:
                    peptide=seq[p-15:p+16]
                if len(seq)-(p+1)<15:
                    peptide+= 'X'*(p+16-len(seq))
                if p+1 in khib:
                    f1.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
                else:
                    f2.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
                p=seq.find('K',p+1)
        t1=line.index('|')
        t2=line.index('|',t1+1)
        pid=line[t1+1:t2]
        n=n+1
        seq=''
        
    else:
        seq=seq+line.strip()

khib=sitepep[pid]
p=seq.find('K')
while(p!=-1):
        if p-15<0:
            peptide='X' * (15-p) + seq[0:p+16]
        else:
            peptide=seq[p-15:p+16]
        if len(seq)-(p+1)<15:
            peptide+= 'X'*(p+16-len(seq))
        if p+1 in khib:
            f1.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
        else:
            f2.write('>{}\t{}\n{}\n'.format(pid,p+1,peptide))
        p=seq.find('K',p+1)
f1.close()
f2.close()
f.close()