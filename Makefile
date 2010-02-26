hmm : hmmus.o hmmguts.o
	gcc hmmus.o hmmguts.o -o hmm

hmmus.o : hmmus.c
	gcc -c hmmus.c

hmmguts.o : hmmguts/hmmguts.c
	gcc -c hmmguts/hmmguts.c

clean :
	rm hmmus.o hmmguts.o
