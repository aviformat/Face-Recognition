import csv

fr=open('file.csv','ab')
i=0
for i in range(170):
    fr.write('./data/abhishek/'+str(i)+'.pgm'+',')
    fr.write(str(1))
    fr.write('\n')
